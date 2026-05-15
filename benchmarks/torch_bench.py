"""
torch_bench.py - 单独测 PyTorch reference 的速度.

为什么单独写这个脚本而不放进 bench_kernel.py:
    1. Reference 极慢, 跑一次 S4 要几十秒, 不能像 GPU kernel 那样跑 100 次
    2. Reference 在 CPU 上跑, 不在同一性能赛道, 放进 GPU benchmark 表里会误导
    3. 单独脚本可以专门为 "看慢有多慢" 设计

回答的问题:
    "如果不写 Triton kernel, 直接用 PyTorch 拼出来的 W8A8 算子, 会有多慢?"

输出:
    - Reference 在每个形状上的延迟 (毫秒级)
    - Triton autotuned 在同样形状上的延迟 (微秒级)
    - 加速比 (Triton 比 reference 快多少倍, 通常是 1000x-10000x 量级)

执行:
    python benchmarks/torch_bench.py            # 默认: 跑 S1, S2 (快), 跳过 S3, S4
    python benchmarks/torch_bench.py --all      # 跑全部 (S3/S4 会很慢, 几分钟)
    python benchmarks/torch_bench.py --shape S2 # 只跑 S2
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from reference.torch_reference import (
    w8a8_scaled_mm_reference,
    quantize_per_token,
    quantize_per_channel,
)
from triton_kernel.w8a8_autotuned import w8a8_scaled_mm_triton as w8a8_autotuned


SHAPES = {
    "S1": (1, 4096, 4096),
    "S2": (16, 4096, 4096),
    "S3": (512, 4096, 4096),
    "S4": (2048, 4096, 4096),
}

# 默认跳过的慢形状 (除非 --all)
SLOW_SHAPES = {"S3", "S4"}


def benchmark_reference(fn, repeat: int) -> dict:
    """
    跑 reference, 用 wall-clock 时间测 (Python time.time).

    为什么不用 torch.cuda.Event:
        reference 大部分时间在 CPU 上跑 (CPU int32 matmul + 来回搬数据)
        cuda.Event 只测 GPU 时间, 会漏掉 CPU 时间
        Python time.time 测 wall-clock, 包含 CPU + GPU + 传输

    为什么 repeat 比 GPU benchmark 少:
        单次跑可能几秒到几十秒, 跑 100 次太慢
    """
    times = []
    for _ in range(repeat):
        t0 = time.time()
        _ = fn()
        torch.cuda.synchronize()   # 确保 GPU 部分也完成
        t1 = time.time()
        times.append((t1 - t0) * 1000)   # ms

    times.sort()
    return {
        "latency_ms": times[len(times) // 2],
        "latency_ms_min": times[0],
        "latency_ms_max": times[-1],
        "n_runs": repeat,
    }


def benchmark_triton(fn, warmup: int = 10, repeat: int = 100) -> dict:
    """跑 Triton, 用 cuda.Event 严格测 GPU 时间."""
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
    return {"latency_ms": times[len(times) // 2]}


def run_shape(shape_id: str, M: int, N: int, K: int, is_slow: bool) -> dict:
    torch.manual_seed(42)

    x_fp16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    # ---- Reference (慢) ----
    # 大形状只跑 3 次, 小形状跑 10 次
    repeat = 3 if is_slow else 10
    print(f"  Running reference ({repeat} runs, each may take seconds)...", flush=True)
    t_start = time.time()
    ref_bench = benchmark_reference(
        lambda: w8a8_scaled_mm_reference(x_q, w_q, sa, sb),
        repeat=repeat,
    )
    t_total = time.time() - t_start
    print(f"    reference: median={ref_bench['latency_ms']:.1f}ms "
          f"(min={ref_bench['latency_ms_min']:.1f}, max={ref_bench['latency_ms_max']:.1f}), "
          f"total time={t_total:.1f}s")

    # ---- Triton autotuned (快) ----
    print(f"  Running Triton autotuned...", flush=True)
    tri_bench = benchmark_triton(lambda: w8a8_autotuned(x_q, w_q, sa, sb))
    print(f"    triton:    median={tri_bench['latency_ms']*1000:.2f}us")

    # ---- 加速比 ----
    speedup = ref_bench["latency_ms"] / tri_bench["latency_ms"]

    return {
        "shape_id": shape_id, "M": M, "N": N, "K": K,
        "reference": ref_bench,
        "triton": tri_bench,
        "speedup": speedup,
    }


def print_results(results: list):
    print()
    print("=" * 110)
    print("torch reference vs Triton autotuned -- 'why we need kernels'")
    print("=" * 110)
    print()

    headers = ["Shape", "(M,N,K)", "Reference latency", "Triton latency", "Speedup"]
    col_w = [6, 22, 25, 20, 20]

    line = ""
    for h, w in zip(headers, col_w):
        line += f"{h:<{w}}"
    print(line)
    print("-" * 110)

    for r in results:
        shape_str = f"({r['M']:>4}, {r['N']:>4}, {r['K']:>4})"
        ref_str = f"{r['reference']['latency_ms']:.1f} ms"
        tri_str = f"{r['triton']['latency_ms']*1000:.2f} us"
        speedup_str = f"{r['speedup']:.0f}x"

        line = (f"{r['shape_id']:<{col_w[0]}}"
                f"{shape_str:<{col_w[1]}}"
                f"{ref_str:<{col_w[2]}}"
                f"{tri_str:<{col_w[3]}}"
                f"{speedup_str:<{col_w[4]}}")
        print(line)

    print()
    print("=" * 110)
    print("解读")
    print("=" * 110)
    print()
    print("Reference 慢的原因:")
    print("  1. PyTorch GPU 不支持 int32 matmul, 数据被搬回 CPU 算")
    print("  2. CPU int32 matmul 是单线程标量循环, 没有 SIMD")
    print("  3. GPU <-> CPU 来回传输有 PCIe 开销")
    print()
    print("Triton 快的原因:")
    print("  1. 全程在 GPU, 不出显存")
    print("  2. 用 Tensor Core int8 mma 指令, 一次算 16x8x32=4096 个乘加")
    print("  3. 数千个 program 并行")
    print()
    print("这就是为什么 kernel 工程存在 - 用 1000x+ 的工程努力, 换 1000x+ 的性能.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=list(SHAPES.keys()) + ["all"], default=None)
    parser.add_argument("--all", action="store_true", help="跑全部形状 (S3/S4 很慢)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # 决定跑哪些形状
    if args.shape and args.shape != "all":
        shapes_to_run = [(args.shape, SHAPES[args.shape])]
    elif args.all or args.shape == "all":
        shapes_to_run = list(SHAPES.items())
    else:
        # 默认: 只跑快形状
        shapes_to_run = [(sid, SHAPES[sid]) for sid in SHAPES if sid not in SLOW_SHAPES]
        print(f"Default: only running fast shapes (S1, S2). Use --all for all shapes.")
        print()

    results = []
    for shape_id, (M, N, K) in shapes_to_run:
        is_slow = shape_id in SLOW_SHAPES
        print(f"=== {shape_id} (M={M}, N={N}, K={K}) ===")
        if is_slow:
            print(f"  WARN: This shape is SLOW (reference may take 30+ seconds per run)")
        r = run_shape(shape_id, M, N, K, is_slow)
        results.append(r)
        print()

    print_results(results)


if __name__ == "__main__":
    main()