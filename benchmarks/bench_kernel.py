"""
Performance benchmark - W8A8 Triton kernel vs cuBLAS fp16 baseline.

设计原则 (四个新手坑的修正):
    1. Warmup: 跑 10 次丢弃, 排除 JIT 编译 + cache 加载时间
    2. Sync:   用 torch.cuda.Event 严格测 kernel 时间
    3. Repeat: 跑 100 次取中位数 (不是平均, 排除 outlier)
    4. Baseline: 对比 cuBLAS fp16 + Roofline 上限, 给出"快/慢"的绝对判断

输出三个核心指标:
    - latency: 单次 kernel 时间 (微秒)
    - TFLOPS:  实际算力利用 (compute intensity)
    - 加速比:   vs cuBLAS fp16

执行:
    python benchmarks/bench_kernel.py                  # 所有形状
    python benchmarks/bench_kernel.py --shape S2       # 只跑 S2
    python benchmarks/bench_kernel.py --reference      # 也测 reference (极慢)
    python benchmarks/bench_kernel.py --detail         # 打印详细 metrics
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from reference.torch_reference import (
    w8a8_scaled_mm_reference,
    cublas_fp16_baseline,
    quantize_per_token,
    quantize_per_channel,
)
from triton_kernel.w8a8_mm import w8a8_scaled_mm_triton


# ============================================================
# 硬件常数 (需要根据实际 GPU 调整)
# ============================================================

HARDWARE_SPECS = {
    "NVIDIA GeForce RTX 4090": {
        "int8_tops_dense": 330e12,
        "fp16_tflops": 165e12,
        "hbm_bandwidth": 1008e9,
    },
    "NVIDIA GeForce RTX 5090": {
        # Triton 当前把 SM12x 当 SM80 用, 实际可达算力打折
        "int8_tops_dense": 838e12,
        "fp16_tflops": 419e12,
        "hbm_bandwidth": 1792e9,
    },
}

DEFAULT_SPEC = HARDWARE_SPECS["NVIDIA GeForce RTX 4090"]


# ============================================================
# LLaMA-7B 测试形状
# ============================================================
SHAPES = {
    "S1": (1, 4096, 4096),
    "S2": (16, 4096, 4096),
    "S3": (512, 4096, 4096),
    "S4": (2048, 4096, 4096),
}


# ============================================================
# 核心计时函数
# ============================================================

def benchmark_fn(fn, warmup: int = 10, repeat: int = 100) -> dict:
    """
    严格的 CUDA kernel 计时.
    """
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
    median_ms = times[len(times) // 2]
    min_ms = times[0]
    p95_ms = times[int(len(times) * 0.95)]

    return {
        "latency_us": median_ms * 1000,
        "latency_us_min": min_ms * 1000,
        "latency_us_p95": p95_ms * 1000,
    }


def compute_metrics(latency_us: float, M: int, N: int, K: int,
                    spec: dict) -> dict:
    """根据延迟和形状, 算 TFLOPS, 带宽利用, Roofline 位置."""
    latency_s = latency_us * 1e-6

    flops = 2 * M * N * K
    tflops = flops / latency_s / 1e12

    bytes_moved = M * K + K * N + M * N * 2
    bandwidth_gbps = bytes_moved / latency_s / 1e9

    ai = flops / bytes_moved
    ai_crossover = spec["int8_tops_dense"] / spec["hbm_bandwidth"]
    if ai < ai_crossover:
        roof_tflops = (spec["hbm_bandwidth"] * ai) / 1e12
        bound = "memory"
    else:
        roof_tflops = spec["int8_tops_dense"] / 1e12
        bound = "compute"

    bandwidth_util = bandwidth_gbps / (spec["hbm_bandwidth"] / 1e9)
    compute_util = (tflops * 1e12) / spec["int8_tops_dense"]
    roof_util = tflops / roof_tflops if roof_tflops > 0 else 0

    return {
        "tflops": tflops,
        "bandwidth_gbps": bandwidth_gbps,
        "bandwidth_util": bandwidth_util,
        "compute_util": compute_util,
        "ai": ai,
        "bound": bound,
        "roof_tflops": roof_tflops,
        "roof_util": roof_util,
    }


# ============================================================
# 单形状测试
# ============================================================

def run_shape(shape_id: str, M: int, N: int, K: int, spec: dict,
              test_reference: bool = False) -> dict:
    torch.manual_seed(42)

    x_fp16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    result = {"shape_id": shape_id, "M": M, "N": N, "K": K}

    # 1. Triton W8A8
    bench = benchmark_fn(lambda: w8a8_scaled_mm_triton(x_q, w_q, sa, sb))
    metrics = compute_metrics(bench["latency_us"], M, N, K, spec)
    result["triton"] = {**bench, **metrics}

    # 2. cuBLAS fp16 baseline
    bench = benchmark_fn(lambda: cublas_fp16_baseline(x_fp16, w_fp16))
    metrics = compute_metrics(bench["latency_us"], M, N, K, spec)
    result["cublas"] = {**bench, **metrics}

    # 3. Reference (可选, 极慢)
    if test_reference and M * N * K < 1e9:
        bench = benchmark_fn(
            lambda: w8a8_scaled_mm_reference(x_q, w_q, sa, sb),
            warmup=2, repeat=5,
        )
        result["reference"] = bench

    # 4. Speedup
    triton_us = result["triton"]["latency_us"]
    cublas_us = result["cublas"]["latency_us"]
    result["speedup_vs_cublas"] = cublas_us / triton_us
    if "reference" in result:
        result["speedup_vs_reference"] = result["reference"]["latency_us"] / triton_us

    return result


# ============================================================
# 打印结果
# ============================================================

def print_results(results: list, spec: dict):
    print()
    print("=" * 120)
    print("Performance Benchmark Results")
    print("=" * 120)
    print()

    print(f"{'Shape':<6} {'(M,N,K)':<22} "
          f"{'Triton (us)':<13} {'cuBLAS (us)':<13} "
          f"{'Speedup':<10} {'TFLOPS':<10} "
          f"{'Roof%':<8} {'Bound':<10}")
    print("-" * 120)

    for r in results:
        shape_str = f"({r['M']:>4}, {r['N']:>4}, {r['K']:>4})"
        triton_us = r["triton"]["latency_us"]
        cublas_us = r["cublas"]["latency_us"]
        speedup = r["speedup_vs_cublas"]
        tflops = r["triton"]["tflops"]
        roof = r["triton"]["roof_util"]
        bound = r["triton"]["bound"]

        speedup_str = f"{speedup:.2f}x"

        print(f"{r['shape_id']:<6} {shape_str:<22} "
              f"{triton_us:<13.2f} {cublas_us:<13.2f} "
              f"{speedup_str:<10} "
              f"{tflops:<10.2f} "
              f"{roof*100:<7.1f}% "
              f"{bound:<10}")

    print()
    print("=" * 120)
    print("解读")
    print("=" * 120)
    print()
    print("Speedup vs cuBLAS:")
    print("  > 1.0x: Triton W8A8 比 cuBLAS fp16 快 (W8A8 路线在该形状上有价值)")
    print("  ~ 1.0x: 持平 (memory-bound 形状常见, dequant 开销吃掉收益)")
    print("  < 1.0x: 比 cuBLAS 慢 (W8A8 在该形状上没意义, 不如直接用 fp16)")
    print()
    print("Roof%: 距 Roofline 上限多远")
    print("  > 70%: 接近物理极限, 优化空间已小")
    print("  30-70%: 健康")
    print("  < 30%: 严重未达上限, 还有大量优化空间 (典型 v1 表现)")
    print()
    print("Bound:")
    print("  memory:  形状受显存带宽限制, 优化方向 = 减少读写")
    print("  compute: 形状受算力限制, 优化方向 = 提高 Tensor Core 占用率")


def print_per_shape_detail(results: list):
    print()
    print("=" * 120)
    print("Per-shape detail")
    print("=" * 120)

    for r in results:
        print()
        print(f"--- {r['shape_id']} (M={r['M']}, N={r['N']}, K={r['K']}) ---")
        print(f"  AI (ops/byte):       {r['triton']['ai']:.2f}")
        print(f"  Bound type:          {r['triton']['bound']}")
        print(f"  Roof TFLOPS:         {r['triton']['roof_tflops']:.2f}")
        print()
        print(f"  Triton W8A8:")
        print(f"    latency (us):      {r['triton']['latency_us']:.2f} "
              f"(min={r['triton']['latency_us_min']:.2f}, "
              f"p95={r['triton']['latency_us_p95']:.2f})")
        print(f"    TFLOPS:            {r['triton']['tflops']:.2f}")
        print(f"    bandwidth (GB/s):  {r['triton']['bandwidth_gbps']:.1f}")
        print(f"    bandwidth util:    {r['triton']['bandwidth_util']*100:.1f}%")
        print(f"    compute util:      {r['triton']['compute_util']*100:.1f}%")
        print(f"    roof util:         {r['triton']['roof_util']*100:.1f}%")
        print()
        print(f"  cuBLAS fp16:")
        print(f"    latency (us):      {r['cublas']['latency_us']:.2f}")
        print(f"    TFLOPS:            {r['cublas']['tflops']:.2f}")
        print(f"    bandwidth (GB/s):  {r['cublas']['bandwidth_gbps']:.1f}")
        print()
        print(f"  Speedup vs cuBLAS:   {r['speedup_vs_cublas']:.2f}x")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=list(SHAPES.keys()) + ["all"],
                        default="all")
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    spec = HARDWARE_SPECS.get(gpu_name, DEFAULT_SPEC)
    print(f"GPU:                 {gpu_name}")
    print(f"INT8 TOPS (dense):   {spec['int8_tops_dense']/1e12:.0f}")
    print(f"FP16 TFLOPS:         {spec['fp16_tflops']/1e12:.0f}")
    print(f"HBM Bandwidth:       {spec['hbm_bandwidth']/1e9:.0f} GB/s")
    print(f"AI Crossover:        {spec['int8_tops_dense']/spec['hbm_bandwidth']:.1f} ops/byte")
    if gpu_name not in HARDWARE_SPECS:
        print(f"WARN: GPU not in spec table, using 4090 numbers as fallback")

    if args.shape == "all":
        shapes_to_run = list(SHAPES.items())
    else:
        shapes_to_run = [(args.shape, SHAPES[args.shape])]

    print()
    results = []
    for shape_id, (M, N, K) in shapes_to_run:
        print(f"Running {shape_id} (M={M}, N={N}, K={K}) ...", end=" ", flush=True)
        t0 = time.time()
        r = run_shape(shape_id, M, N, K, spec, test_reference=args.reference)
        t1 = time.time()
        print(f"done ({t1-t0:.1f}s)")
        results.append(r)

    print_results(results, spec)
    if args.detail:
        print_per_shape_detail(results)


if __name__ == "__main__":
    main()