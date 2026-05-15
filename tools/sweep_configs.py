"""
sweep_configs.py - 暴力网格扫描,找指定形状上的真正最优 BLOCK 配置.

为什么不直接加候选到 autotune:
    凭直觉加候选可能加错方向. 先用网格扫描确认"什么配置真的好",
    再把表现最好的几个加到 autotune 候选列表.

扫描空间 (针对小 M 形状优化):
    BLOCK_M:    16, 32, 64
    BLOCK_N:    32, 64, 128
    BLOCK_K:    32, 64, 128, 256
    num_warps:  2, 4
    num_stages: 2, 3
    
    共 3×3×4×2×2 = 144 个配置 (约 2-3 分钟)

执行:
    python tools/sweep_configs.py --shape S2       # 默认扫 S2
    python tools/sweep_configs.py --shape S1
    python tools/sweep_configs.py --shape S2 --top 10  # 显示 top 10
"""

import argparse
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).parent.parent))
from reference.torch_reference import quantize_per_token, quantize_per_channel
from triton_kernel.w8a8_autotuned import _w8a8_scaled_mm_kernel_autotuned


SHAPES = {
    "S1": (1, 4096, 4096),
    "S2": (16, 4096, 4096),
    "S3": (512, 4096, 4096),
    "S4": (2048, 4096, 4096),
}


# 扫描空间 - 针对小 M 优化, 大 BLOCK 上次扫过了
SWEEP_BLOCK_M = [16, 32, 64]
SWEEP_BLOCK_N = [32, 64, 128]
SWEEP_BLOCK_K = [32, 64, 128, 256]
SWEEP_NUM_WARPS = [2, 4]
SWEEP_NUM_STAGES = [2, 3]


def generate_configs():
    """生成所有候选配置."""
    configs = []
    for bm in SWEEP_BLOCK_M:
        for bn in SWEEP_BLOCK_N:
            for bk in SWEEP_BLOCK_K:
                for nw in SWEEP_NUM_WARPS:
                    for ns in SWEEP_NUM_STAGES:
                        configs.append(triton.Config(
                            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                            num_warps=nw, num_stages=ns,
                        ))
    return configs


def time_kernel_with_config(x_q, w_q, sa, sb, config, repeat=50):
    """直接调 jit kernel, 测指定 config 的延迟."""
    M, K = x_q.shape
    _, N = w_q.shape

    c = torch.empty((M, N), dtype=torch.float16, device="cuda")

    BLOCK_M = config.kwargs["BLOCK_M"]
    BLOCK_N = config.kwargs["BLOCK_N"]
    BLOCK_K = config.kwargs["BLOCK_K"]
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    def run():
        _w8a8_scaled_mm_kernel_autotuned.fn[grid](
            x_q, w_q, c,
            sa, sb,   # 注意: 直接传 2D scale (匹配新 wrapper)
            M, N, K,
            x_q.stride(0), x_q.stride(1),
            w_q.stride(0), w_q.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )

    try:
        # warmup
        for _ in range(20):
            run()
        torch.cuda.synchronize()

        # 测
        times = []
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times.sort()
        return times[len(times) // 2] * 1000   # us
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=list(SHAPES.keys()), default="S2")
    parser.add_argument("--top", type=int, default=15,
                        help="显示 top N 个最快配置")
    args = parser.parse_args()

    M, N, K = SHAPES[args.shape]
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Shape: {args.shape} (M={M}, N={N}, K={K})")

    configs = generate_configs()
    print(f"Sweeping {len(configs)} configurations...")
    print(f"  BLOCK_M: {SWEEP_BLOCK_M}")
    print(f"  BLOCK_N: {SWEEP_BLOCK_N}")
    print(f"  BLOCK_K: {SWEEP_BLOCK_K}")
    print(f"  num_warps: {SWEEP_NUM_WARPS}")
    print(f"  num_stages: {SWEEP_NUM_STAGES}")
    print()

    # 准备数据
    torch.manual_seed(42)
    x_fp16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    # 扫描
    print(f"Running sweep (this takes ~2-3 minutes)...")
    print()
    results = []

    for i, config in enumerate(configs):
        latency = time_kernel_with_config(x_q, w_q, sa, sb, config)
        bm = config.kwargs["BLOCK_M"]
        bn = config.kwargs["BLOCK_N"]
        bk = config.kwargs["BLOCK_K"]
        nw = config.num_warps
        ns = config.num_stages
        cfg_str = f"BM={bm:>3}, BN={bn:>3}, BK={bk:>3}, w={nw}, s={ns}"

        if latency is None:
            status = "FAILED"
        else:
            status = f"{latency:>6.2f} us"

        # 简洁进度: 每 20 个打一行进度
        if (i + 1) % 20 == 0 or i == len(configs) - 1:
            print(f"  [{i+1:>3}/{len(configs)}] last: {cfg_str} -> {status}")

        if latency is not None:
            results.append({
                "config_str": cfg_str,
                "latency": latency,
                "bm": bm, "bn": bn, "bk": bk,
                "nw": nw, "ns": ns,
            })

    # 排序展示
    results.sort(key=lambda x: x["latency"])

    print()
    print("=" * 90)
    print(f"Top {args.top} fastest configurations on {args.shape} (M={M}, N={N}, K={K})")
    print("=" * 90)
    print()
    print(f"{'Rank':<6} {'Latency (us)':<14} {'Config':<55}")
    print("-" * 90)
    for rank, r in enumerate(results[:args.top]):
        marker = "  <-- TRUE BEST" if rank == 0 else ""
        print(f"{rank+1:<6} {r['latency']:<14.2f} {r['config_str']:<55}{marker}")

    # 分析当前 autotune 候选里哪些没覆盖
    print()
    print("=" * 90)
    print("分析: 这些 top configs 在当前 autotune 候选里吗?")
    print("=" * 90)

    from triton_kernel.w8a8_autotuned import _AUTOTUNE_CONFIGS
    current_configs = set()
    for c in _AUTOTUNE_CONFIGS:
        current_configs.add((c.kwargs["BLOCK_M"], c.kwargs["BLOCK_N"],
                             c.kwargs["BLOCK_K"], c.num_warps, c.num_stages))

    print()
    print(f"{'Rank':<6} {'Config':<55} {'In autotune?':<15}")
    print("-" * 90)
    for rank, r in enumerate(results[:args.top]):
        key = (r["bm"], r["bn"], r["bk"], r["nw"], r["ns"])
        in_set = key in current_configs
        status = "Yes" if in_set else "NO (missing!)"
        print(f"{rank+1:<6} {r['config_str']:<55} {status:<15}")

    # 建议
    print()
    print("=" * 90)
    print("建议添加到 autotune 候选的配置")
    print("=" * 90)
    print()
    missing = []
    for r in results[:args.top]:
        key = (r["bm"], r["bn"], r["bk"], r["nw"], r["ns"])
        if key not in current_configs:
            missing.append(r)

    if not missing:
        print("Top configs 已都在 autotune 候选里. 问题不在候选缺失.")
    else:
        print(f"建议加入以下 {len(missing)} 个配置 (Triton Config 格式):")
        print()
        for r in missing:
            print(f'    triton.Config(')
            print(f'        {{"BLOCK_M": {r["bm"]}, "BLOCK_N": {r["bn"]}, "BLOCK_K": {r["bk"]}}},')
            print(f'        num_warps={r["nw"]}, num_stages={r["ns"]},')
            print(f'    ),  # {r["latency"]:.2f}us')


if __name__ == "__main__":
    main()