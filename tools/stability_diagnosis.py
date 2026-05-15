"""
stability_diagnosis.py - 验证 benchmark 跨次执行的稳定性.

回答的问题:
    "同一个 kernel 在同一台机器上, 跨次执行的真实抖动是多少?"

为什么需要这个:
    之前几次 benchmark 数据互相对照, 发现 naive S1 从 148us -> 53us, 差 95us.
    我们没动 naive 代码, 这种巨大差异只能来自 benchmark 环境抖动.
    如果跨次抖动 > 我们优化收益, 那所有"性能改善"判断都不可信.

四个层次的稳定性测试:
    Layer 1: GPU 热起来后, 同一次 benchmark 内的稳定性
    Layer 2: 同一次脚本内, 重复测多次的稳定性
    Layer 3: 不同 Python 进程间的稳定性 (用户手动多跑几次脚本对比)
    Layer 4: 删 Triton cache 后的稳定性 (是否 JIT 抖动是主因)

执行:
    python tools/stability_diagnosis.py
    python tools/stability_diagnosis.py --clear-cache  # 删 Triton cache 后测
    python tools/stability_diagnosis.py --shape S2     # 指定形状
"""

import argparse
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from reference.torch_reference import (
    quantize_per_token,
    quantize_per_channel,
    cublas_fp16_baseline,
)
from triton_kernel.w8a8_naive import w8a8_scaled_mm_triton as w8a8_naive
from triton_kernel.w8a8_autotuned import w8a8_scaled_mm_triton as w8a8_autotuned


SHAPES = {
    "S1": (1, 4096, 4096),
    "S2": (16, 4096, 4096),
    "S3": (512, 4096, 4096),
    "S4": (2048, 4096, 4096),
}


# ============================================================
# 激进 warmup: 让 GPU 充分热起来
# ============================================================

def aggressive_warmup(fn, duration_s: float = 2.0):
    """
    持续调用 fn 直到 duration_s 秒过去.

    目的: 让 GPU 进入稳定时钟状态.
    typical GPU boost 需要 1-2 秒 sustained workload 才稳定.
    比单纯跑 100 次 warmup 更可靠.
    """
    t_start = time.time()
    n = 0
    while time.time() - t_start < duration_s:
        fn()
        n += 1
    torch.cuda.synchronize()
    return n


# ============================================================
# 单次 benchmark (带高质量 warmup)
# ============================================================

def benchmark_once(fn, repeat: int = 100, warmup_s: float = 1.0) -> float:
    """跑一次完整 benchmark, 返回 latency 中位数 (us)."""
    # 激进 warmup
    aggressive_warmup(fn, warmup_s)

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
    return times[len(times) // 2] * 1000   # us


# ============================================================
# Layer 1+2: 同一脚本内多次 benchmark, 看跨"轮"抖动
# ============================================================

def stability_test(label: str, fn, n_rounds: int = 5,
                   repeat: int = 100, warmup_s: float = 1.0) -> dict:
    """
    跑 n_rounds 轮独立 benchmark, 每轮内部 repeat 次取中位.

    返回:
        - rounds: 每轮的 median 列表
        - overall_median: 跨轮 median 的 median (双层防抖)
        - cross_round_p2p: 跨轮峰峰值 (max - min), 衡量"跨次抖动"
        - cross_round_stddev: 跨轮标准差
    """
    print(f"\n  Testing: {label}")
    print(f"    {n_rounds} rounds × {repeat} runs each, with {warmup_s}s aggressive warmup per round")

    rounds = []
    for r in range(n_rounds):
        # 每轮独立 warmup, 模拟"程序冷启动"
        t = benchmark_once(fn, repeat=repeat, warmup_s=warmup_s)
        rounds.append(t)
        print(f"    Round {r+1}: median = {t:.2f} us")

    overall_median = statistics.median(rounds)
    cross_round_p2p = max(rounds) - min(rounds)
    cross_round_stddev = statistics.stdev(rounds) if len(rounds) > 1 else 0.0

    return {
        "label": label,
        "rounds": rounds,
        "overall_median": overall_median,
        "min": min(rounds),
        "max": max(rounds),
        "cross_round_p2p": cross_round_p2p,
        "cross_round_stddev": cross_round_stddev,
        "cross_round_cv": cross_round_stddev / overall_median if overall_median > 0 else 0,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=list(SHAPES.keys()), default="S2",
                        help="要测的形状")
    parser.add_argument("--clear-cache", action="store_true",
                        help="测试前清除 Triton JIT cache, 看 JIT 是否影响稳定性")
    parser.add_argument("--rounds", type=int, default=5,
                        help="跑几轮独立 benchmark")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    print("=" * 80)
    print("Benchmark Stability Diagnosis")
    print("=" * 80)
    print(f"GPU:         {torch.cuda.get_device_name(0)}")
    print(f"Shape:       {args.shape} (M={SHAPES[args.shape][0]}, "
          f"N={SHAPES[args.shape][1]}, K={SHAPES[args.shape][2]})")
    print(f"Rounds:      {args.rounds}")
    print()

    # 可选: 清除 Triton cache
    if args.clear_cache:
        cache_dir = Path.home() / ".triton" / "cache"
        if cache_dir.exists():
            print(f"Clearing Triton cache at {cache_dir} ...")
            shutil.rmtree(cache_dir)
            print("  Cache cleared.")
        else:
            print(f"  No cache at {cache_dir}")
        print()

    # 准备数据
    M, N, K = SHAPES[args.shape]
    torch.manual_seed(42)
    x_fp16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    # 三个对手都测稳定性
    results = []

    # 1. cuBLAS fp16 (锚点 - 我们没动过的对照)
    r = stability_test(
        "cuBLAS fp16 (anchor)",
        lambda: cublas_fp16_baseline(x_fp16, w_fp16),
        n_rounds=args.rounds,
    )
    results.append(r)

    # 2. Triton naive (我们没动过, 应该和上次差不多)
    r = stability_test(
        "Triton naive (v1, unchanged)",
        lambda: w8a8_naive(x_q, w_q, sa, sb),
        n_rounds=args.rounds,
    )
    results.append(r)

    # 3. Triton autotuned (wrapper 优化版)
    r = stability_test(
        "Triton autotuned (v2 + wrapper fix)",
        lambda: w8a8_autotuned(x_q, w_q, sa, sb),
        n_rounds=args.rounds,
    )
    results.append(r)

    # 报告
    print()
    print("=" * 80)
    print("Stability Report")
    print("=" * 80)
    print()
    headers = ["Implementation", "Median", "Min", "Max", "P2P", "StdDev", "CV%"]
    widths = [38, 10, 10, 10, 10, 10, 8]

    line = ""
    for h, w in zip(headers, widths):
        line += f"{h:<{w}}"
    print(line)
    print("-" * 90)

    for r in results:
        row = [
            r["label"],
            f"{r['overall_median']:.2f}",
            f"{r['min']:.2f}",
            f"{r['max']:.2f}",
            f"{r['cross_round_p2p']:.2f}",
            f"{r['cross_round_stddev']:.2f}",
            f"{r['cross_round_cv']*100:.1f}%",
        ]
        line = ""
        for v, w in zip(row, widths):
            line += f"{v:<{w}}"
        print(line)

    print()
    print("Legend:")
    print("  Median:  跨轮 median 的 median (主要数据点)")
    print("  P2P:     Peak-to-peak (max - min), 跨次抖动幅度")
    print("  CV%:     Coefficient of Variation = stddev / median, 相对抖动")
    print()
    print("Stability rating:")
    print("  CV < 2%:   Excellent  - 数据高度可信")
    print("  CV 2-5%:   Good       - 数据可信, 5us 内的差异需谨慎判断")
    print("  CV 5-10%:  Marginal   - 10us 内的差异不可信")
    print("  CV > 10%:  Poor       - 跨次抖动太大, 优化判断不可靠")
    print()

    # 自动诊断
    cv_anchor = results[0]["cross_round_cv"] * 100   # cuBLAS 是锚点

    print("=" * 80)
    print("Diagnosis")
    print("=" * 80)
    print()

    if cv_anchor < 2:
        print(f"cuBLAS CV = {cv_anchor:.1f}% (Excellent)")
        print("  -> GPU/系统层面稳定")
        print("  -> 如果 Triton CV 也低, 之前的 benchmark 数据可信")
        print("  -> 如果 Triton CV 高, 问题在 Triton/wrapper, 不在系统")
    elif cv_anchor < 5:
        print(f"cuBLAS CV = {cv_anchor:.1f}% (Good)")
        print("  -> GPU/系统抖动可接受")
        print("  -> 优化收益 > 2x 系统抖动 才能算真实进步")
    elif cv_anchor < 10:
        print(f"cuBLAS CV = {cv_anchor:.1f}% (Marginal)")
        print("  -> 系统抖动较大 (RunPod 共享硬件影响)")
        print("  -> 只有 > 10% 的差异才算真实优化")
    else:
        print(f"cuBLAS CV = {cv_anchor:.1f}% (Poor)")
        print("  -> 系统层面非常不稳定, 这次测量本身不可信")
        print("  -> 建议: 换实例 / 等会儿再测 / 接受不确定性")

    # 给出"可信的优化阈值"建议
    overall_anchor = results[0]["overall_median"]
    p2p_anchor = results[0]["cross_round_p2p"]
    print()
    print(f"基于 cuBLAS 锚点 (median={overall_anchor:.2f}us, p2p={p2p_anchor:.2f}us):")
    print(f"  -> 真实优化必须 > {p2p_anchor*1.5:.1f}us 才能确认 (1.5x p2p)")
    print(f"  -> 这意味着 v1 vs v2 差 < {p2p_anchor*1.5:.1f}us 的差异都可能是抖动")


if __name__ == "__main__":
    main()