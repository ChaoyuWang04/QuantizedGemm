"""
diagnose_autotune.py - 诊断 S2 异常 (autotuned 比 naive 慢) 的工具.

四个诊断步骤 (按"消除不确定性"原则排序):

  Step 1: 暴露 autotune 选了哪个配置
    -> 验证假设 H1 (autotune 选错)

  Step 2: 跑多次 benchmark 看抖动分布
    -> 验证假设 H2 (测量噪声)

  Step 3: 手动暴力测每个候选配置, 找真实 best
    -> 验证假设 H3 (候选不全) + H5 (代码 bug)

  Step 4: 对比"autotune 的 best" vs "暴力测的 best"
    -> 决定 fix 方向: 重写 autotune 还是扩充候选列表

执行:
    python tools/diagnose_autotune.py --shape S2     # 调查 S2 (默认)
    python tools/diagnose_autotune.py --shape S4     # 也可以查其他形状
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).parent.parent))
from reference.torch_reference import (
    quantize_per_token,
    quantize_per_channel,
    cublas_fp16_baseline,
)
# 直接 import kernel 和 config list, 用来手动测每个配置
from triton_kernel.w8a8_autotuned import (
    _w8a8_scaled_mm_kernel_autotuned,
    _AUTOTUNE_CONFIGS,
    w8a8_scaled_mm_triton as w8a8_autotuned,
)
from triton_kernel.w8a8_naive import w8a8_scaled_mm_triton as w8a8_naive


SHAPES = {
    "S1": (1, 4096, 4096),
    "S2": (16, 4096, 4096),
    "S3": (512, 4096, 4096),
    "S4": (2048, 4096, 4096),
}


# ============================================================
# 通用 benchmark
# ============================================================

def benchmark_fn(fn, warmup: int = 20, repeat: int = 100) -> dict:
    """高质量计时, warmup 多一点, 减少抖动."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return {
        "median": times[len(times) // 2] * 1000,
        "min": times[0] * 1000,
        "p05": times[int(len(times) * 0.05)] * 1000,
        "p95": times[int(len(times) * 0.95)] * 1000,
        "max": times[-1] * 1000,
    }


# ============================================================
# Step 1: 暴露 autotune 选了什么
# ============================================================

def step1_expose_best_config(x_q, w_q, sa, sb):
    print("=" * 80)
    print("Step 1: autotune 选了哪个配置?")
    print("=" * 80)

    # 第一次调用触发 autotune
    print("\n触发 autotune (第一次跑会试所有候选)...")
    t0 = time.time()
    _ = w8a8_autotuned(x_q, w_q, sa, sb)
    torch.cuda.synchronize()
    print(f"autotune 耗时: {time.time()-t0:.2f}s")

    # 从 cache 里读出 best_config
    # Triton autotune 把结果存在 kernel.cache 里
    cache = _w8a8_scaled_mm_kernel_autotuned.cache
    if not cache:
        print("WARN: autotune cache 为空")
        return None

    # cache 是 dict, key 是 autotune key 的 tuple, value 是 best Config
    print(f"\nautotune cache 大小: {len(cache)}")
    for key, best_config in cache.items():
        print(f"  Key (M,N,K): {key}")
        print(f"  Best config: {best_config}")
        return best_config

    return None


# ============================================================
# Step 2: 测量抖动分布
# ============================================================

def step2_measure_noise(x_q, w_q, sa, sb):
    print()
    print("=" * 80)
    print("Step 2: 测量噪声分析")
    print("=" * 80)
    print()
    print("跑 3 轮独立 benchmark, 看中位数是否稳定:")

    for run_idx in range(3):
        # naive
        b_n = benchmark_fn(lambda: w8a8_naive(x_q, w_q, sa, sb))
        # autotuned (已 autotune 完, 这里测真实运行)
        b_a = benchmark_fn(lambda: w8a8_autotuned(x_q, w_q, sa, sb))

        print(f"  Run {run_idx+1}: naive={b_n['median']:.2f}us "
              f"[p05={b_n['p05']:.2f}, p95={b_n['p95']:.2f}], "
              f"autotuned={b_a['median']:.2f}us "
              f"[p05={b_a['p05']:.2f}, p95={b_a['p95']:.2f}]")

    print()
    print("解读: 如果 3 轮 median 差距 < 5us, autotune vs naive 的 ~10us 差距是真实的;")
    print("       如果 3 轮 median 抖动 > 10us, 那 ~10us 差距可能是噪声.")


# ============================================================
# Step 3: 手动暴力测所有候选
# ============================================================

def time_kernel_with_config(x_q, w_q, sa, sb, config: triton.Config) -> float:
    """
    用指定的 config 直接调 kernel, 测 latency.

    这绕过了 autotune 装饰器, 直接传配置, 让我们能逐个测每个候选.
    """
    M, K = x_q.shape
    _, N = w_q.shape

    c = torch.empty((M, N), dtype=torch.float16, device="cuda")
    sa_1d = sa.squeeze(-1).contiguous()
    sb_1d = sb.squeeze(0).contiguous()

    BLOCK_M = config.kwargs["BLOCK_M"]
    BLOCK_N = config.kwargs["BLOCK_N"]
    BLOCK_K = config.kwargs["BLOCK_K"]
    num_warps = config.num_warps
    num_stages = config.num_stages

    # 计算 grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    def run():
        # 直接调 jit 函数, 不走 autotune
        _w8a8_scaled_mm_kernel_autotuned.fn[grid](
            x_q, w_q, c,
            sa_1d, sb_1d,
            M, N, K,
            x_q.stride(0), x_q.stride(1),
            w_q.stride(0), w_q.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    try:
        # warmup
        for _ in range(10):
            run()
        torch.cuda.synchronize()

        # 测 50 次
        times = []
        for _ in range(50):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times.sort()
        return times[len(times) // 2] * 1000   # median us
    except Exception as e:
        return None   # 编译失败或运行失败


def step3_brute_force_all_configs(x_q, w_q, sa, sb):
    print()
    print("=" * 80)
    print("Step 3: 手动暴力测所有候选配置")
    print("=" * 80)
    print()
    print(f"测试 {len(_AUTOTUNE_CONFIGS)} 个候选:")
    print()

    results = []
    for i, config in enumerate(_AUTOTUNE_CONFIGS):
        bm = config.kwargs["BLOCK_M"]
        bn = config.kwargs["BLOCK_N"]
        bk = config.kwargs["BLOCK_K"]
        nw = config.num_warps
        ns = config.num_stages
        cfg_str = f"BLOCK=({bm:>3}, {bn:>3}, {bk:>3}), warps={nw}, stages={ns}"

        latency_us = time_kernel_with_config(x_q, w_q, sa, sb, config)
        if latency_us is None:
            print(f"  [{i:2}] {cfg_str}  -> FAILED (compile or run error)")
            results.append((config, None, cfg_str))
        else:
            print(f"  [{i:2}] {cfg_str}  -> {latency_us:.2f} us")
            results.append((config, latency_us, cfg_str))

    # 排名
    valid = [(c, t, s) for c, t, s in results if t is not None]
    valid.sort(key=lambda x: x[1])

    print()
    print("Top 5 fastest configs (暴力测出来的真实排名):")
    for rank, (c, t, s) in enumerate(valid[:5]):
        marker = " <- TRUE BEST" if rank == 0 else ""
        print(f"  Rank {rank+1}: {t:>6.2f} us  | {s}{marker}")

    return valid[0] if valid else None   # (config, latency, cfg_str)


# ============================================================
# Step 4: 对比 autotune 选的 vs 暴力测的
# ============================================================

def step4_compare(autotune_best, brute_best, x_q, w_q, sa, sb):
    print()
    print("=" * 80)
    print("Step 4: autotune 选的 vs 暴力测的 best")
    print("=" * 80)
    print()

    if autotune_best is None or brute_best is None:
        print("无法对比 (缺数据)")
        return

    # autotune 选的配置, 再单独测一次确认延迟
    autotune_latency = time_kernel_with_config(x_q, w_q, sa, sb, autotune_best)

    brute_config, brute_latency, brute_str = brute_best
    autotune_str = f"BLOCK=({autotune_best.kwargs['BLOCK_M']:>3}, {autotune_best.kwargs['BLOCK_N']:>3}, {autotune_best.kwargs['BLOCK_K']:>3}), warps={autotune_best.num_warps}, stages={autotune_best.num_stages}"

    print(f"autotune chose: {autotune_str}")
    print(f"  -> {autotune_latency:.2f} us")
    print()
    print(f"brute  best:    {brute_str}")
    print(f"  -> {brute_latency:.2f} us")
    print()

    if autotune_latency is None:
        print("autotune 选的配置无法重测, 可能是 autotune 内部问题")
        return

    diff = autotune_latency - brute_latency
    if diff <= 1.0:
        print(f"DIAGNOSIS: autotune 和暴力测一致 (差 {diff:.2f}us, 可忽略)")
        print("   -> 问题不在 autotune, 而在 benchmark 测量噪声 (H2)")
    elif diff < brute_latency * 0.1:
        print(f"DIAGNOSIS: autotune 选得稍差 (差 {diff:.2f}us = {diff/brute_latency*100:.1f}%)")
        print("   -> 小幅差异, 可能 autotune 试 best 时撞了抖动")
    else:
        print(f"DIAGNOSIS: autotune 显著选错 (差 {diff:.2f}us = {diff/brute_latency*100:.1f}%)")
        print("   -> 可能原因: autotune key 不稳, GPU 占用变化, 候选列表问题")
        print("   -> Fix 建议: 给 autotune 加 num_trials > 1, 或固化最优 config")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=list(SHAPES.keys()), default="S2")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    M, N, K = SHAPES[args.shape]
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Investigating shape: {args.shape} (M={M}, N={N}, K={K})")
    print()

    # 准备数据
    torch.manual_seed(42)
    x_fp16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    # Step 1: 暴露 autotune 选了什么
    autotune_best = step1_expose_best_config(x_q, w_q, sa, sb)

    # Step 2: 测噪声
    step2_measure_noise(x_q, w_q, sa, sb)

    # Step 3: 暴力测所有候选
    brute_best = step3_brute_force_all_configs(x_q, w_q, sa, sb)

    # Step 4: 对比
    step4_compare(autotune_best, brute_best, x_q, w_q, sa, sb)

    print()
    print("=" * 80)
    print("总结")
    print("=" * 80)
    print()
    print("如果 Step 4 显示 'autotune 和暴力测一致':")
    print("  -> S2 异常是测量噪声, 实际 autotune 是对的")
    print("  -> 行动: 不用改代码, 但 benchmark 时多跑几次取中位")
    print()
    print("如果 Step 4 显示 'autotune 显著选错':")
    print("  -> autotune 内部有问题")
    print("  -> 行动: 把暴力测出的 best config 固化, 不依赖 autotune")


if __name__ == "__main__":
    main()