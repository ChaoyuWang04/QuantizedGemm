"""
Performance benchmark - W8A8 Triton kernel 多版本对比.

对手矩阵:
    1. Triton W8A8 naive (v1)     - 我们项目第一版, 固定 BLOCK, 当 baseline
    2. Triton W8A8 autotuned (v2) - 我们项目当前最优版
    3. cuBLAS fp16                - 用户的"替代方案", W8A8 路线要打的对手

执行:
    python benchmarks/bench_kernel.py                    # 所有形状
    python benchmarks/bench_kernel.py --shape S2         # 只跑 S2
    python benchmarks/bench_kernel.py --detail           # 每形状详细 metrics

Note: 想跑 PyTorch reference 的极端慢对比, 用 benchmarks/torch_bench.py
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from reference.torch_reference import (
    cublas_fp16_baseline,
    quantize_per_token,
    quantize_per_channel,
)
from triton_kernel.w8a8_naive import w8a8_scaled_mm_triton as w8a8_naive
from triton_kernel.w8a8_autotuned import w8a8_scaled_mm_triton as w8a8_autotuned


# ============================================================
# 硬件常数
# ============================================================
HARDWARE_SPECS = {
    "NVIDIA GeForce RTX 4090": {
        "int8_tops_dense": 330e12,
        "fp16_tflops": 165e12,
        "hbm_bandwidth": 1008e9,
    },
    "NVIDIA GeForce RTX 5090": {
        "int8_tops_dense": 838e12,
        "fp16_tflops": 419e12,
        "hbm_bandwidth": 1792e9,
    },
}
DEFAULT_SPEC = HARDWARE_SPECS["NVIDIA GeForce RTX 4090"]

SHAPES = {
    "S1": (1, 4096, 4096),
    "S2": (16, 4096, 4096),
    "S3": (512, 4096, 4096),
    "S4": (2048, 4096, 4096),
}


# ============================================================
# 核心计时
# ============================================================

def benchmark_fn(fn, warmup: int = 10, repeat: int = 100) -> dict:
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
        "latency_us": times[len(times) // 2] * 1000,
        "latency_us_min": times[0] * 1000,
        "latency_us_p95": times[int(len(times) * 0.95)] * 1000,
    }


def compute_metrics(latency_us: float, M: int, N: int, K: int,
                    spec: dict) -> dict:
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

    return {
        "tflops": tflops,
        "bandwidth_gbps": bandwidth_gbps,
        "bandwidth_util": bandwidth_gbps / (spec["hbm_bandwidth"] / 1e9),
        "compute_util": (tflops * 1e12) / spec["int8_tops_dense"],
        "ai": ai,
        "bound": bound,
        "roof_tflops": roof_tflops,
        "roof_util": tflops / roof_tflops if roof_tflops > 0 else 0,
    }


# ============================================================
# 单形状测试
# ============================================================

def run_shape(shape_id: str, M: int, N: int, K: int, spec: dict) -> dict:
    torch.manual_seed(42)

    x_fp16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    result = {"shape_id": shape_id, "M": M, "N": N, "K": K}

    # 1. Triton naive (v1)
    bench = benchmark_fn(lambda: w8a8_naive(x_q, w_q, sa, sb))
    result["triton_naive"] = {**bench, **compute_metrics(bench["latency_us"], M, N, K, spec)}

    # 2. Triton autotuned (v2)
    bench = benchmark_fn(lambda: w8a8_autotuned(x_q, w_q, sa, sb))
    result["triton_autotuned"] = {**bench, **compute_metrics(bench["latency_us"], M, N, K, spec)}

    # 3. cuBLAS fp16
    bench = benchmark_fn(lambda: cublas_fp16_baseline(x_fp16, w_fp16))
    result["cublas_fp16"] = {**bench, **compute_metrics(bench["latency_us"], M, N, K, spec)}

    # Speedup (基准: autotuned)
    base = result["triton_autotuned"]["latency_us"]
    result["speedup"] = {
        "naive_vs_autotuned": result["triton_naive"]["latency_us"] / base,
        "autotuned_vs_cublas": result["cublas_fp16"]["latency_us"] / base,
    }

    return result


# ============================================================
# 打印
# ============================================================

def print_main_table(results: list):
    print()
    print("=" * 110)
    print("Performance Benchmark - W8A8 Triton vs cuBLAS fp16 (latency in microseconds)")
    print("=" * 110)
    print()

    headers = ["Shape", "(M,N,K)", "naive", "autotuned", "cuBLAS fp16",
               "TFLOPS", "Roof%", "Bound"]
    col_w = [6, 22, 11, 12, 13, 12, 8, 10]

    line = ""
    for h, w in zip(headers, col_w):
        line += f"{h:<{w}}"
    print(line)
    print("-" * 110)

    for r in results:
        shape_str = f"({r['M']:>4}, {r['N']:>4}, {r['K']:>4})"
        row = [
            r["shape_id"], shape_str,
            f"{r['triton_naive']['latency_us']:.2f}",
            f"{r['triton_autotuned']['latency_us']:.2f}",
            f"{r['cublas_fp16']['latency_us']:.2f}",
            f"{r['triton_autotuned']['tflops']:.2f}",
            f"{r['triton_autotuned']['roof_util']*100:.1f}%",
            r["triton_autotuned"]["bound"],
        ]
        line = ""
        for v, w in zip(row, col_w):
            line += f"{v:<{w}}"
        print(line)


def print_speedup_table(results: list):
    print()
    print("=" * 110)
    print("Speedup Analysis (基准: Triton autotuned, 数字>1 表示 autotuned 更快)")
    print("=" * 110)
    print()

    headers = ["Shape", "naive -> autotuned (优化进度)", "autotuned vs cuBLAS fp16 (替代方案)"]
    col_w = [6, 40, 40]

    line = ""
    for h, w in zip(headers, col_w):
        line += f"{h:<{w}}"
    print(line)
    print("-" * 110)

    for r in results:
        sp = r["speedup"]
        x = sp["naive_vs_autotuned"]
        col2 = f"{x:.2f}x" + (" faster than naive" if x > 1 else " slower than naive")

        y = sp["autotuned_vs_cublas"]
        col3 = f"{y:.2f}x" + (" faster than cuBLAS" if y > 1 else " slower than cuBLAS")

        line = f"{r['shape_id']:<{col_w[0]}}{col2:<{col_w[1]}}{col3:<{col_w[2]}}"
        print(line)


def print_legend():
    print()
    print("=" * 110)
    print("解读")
    print("=" * 110)
    print()
    print("对手身份:")
    print("  naive (v1)     = 我们第一版, 固定 BLOCK=64, 当 baseline")
    print("  autotuned (v2) = 我们当前最优版, autotune 自动选 BLOCK")
    print("  cuBLAS fp16    = torch.matmul fp16, 用户的'替代方案'")
    print()
    print("怎么读:")
    print("  naive -> autotuned > 1x: 优化有效 (autotune 比固定 BLOCK 快)")
    print("  autotuned vs cuBLAS > 1x: 用 W8A8 比 fp16 快, 路线有价值")
    print("  autotuned vs cuBLAS < 1x: 在该形状上 W8A8 不如 fp16, 路线无意义")
    print()
    print("  Roof%: 距 Roofline 物理上限的百分比")
    print("    > 70%: 接近极限, 优化空间小")
    print("    30-70%: 健康范围")
    print("    < 30%: 还有大量优化空间")


def print_per_shape_detail(results: list):
    print()
    print("=" * 110)
    print("Per-shape detail (Triton autotuned)")
    print("=" * 110)

    for r in results:
        a = r["triton_autotuned"]
        print()
        print(f"--- {r['shape_id']} (M={r['M']}, N={r['N']}, K={r['K']}) ---")
        print(f"  AI:             {a['ai']:.2f} ops/byte ({a['bound']}-bound)")
        print(f"  Roof TFLOPS:    {a['roof_tflops']:.2f}")
        print(f"  Autotuned: lat={a['latency_us']:.2f}us "
              f"(min={a['latency_us_min']:.2f}, p95={a['latency_us_p95']:.2f}), "
              f"TFLOPS={a['tflops']:.2f}, "
              f"BW util={a['bandwidth_util']*100:.1f}%, "
              f"Compute util={a['compute_util']*100:.1f}%, "
              f"Roof util={a['roof_util']*100:.1f}%")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=list(SHAPES.keys()) + ["all"], default="all")
    parser.add_argument("--detail", action="store_true")
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
        print(f"WARN: GPU not in spec table, using 4090 fallback")

    if args.shape == "all":
        shapes_to_run = list(SHAPES.items())
    else:
        shapes_to_run = [(args.shape, SHAPES[args.shape])]

    print()
    print("Note: 第一次跑 autotuned 版本会触发 autotune (每形状 10-30s)")
    print()

    results = []
    for shape_id, (M, N, K) in shapes_to_run:
        print(f"Running {shape_id} (M={M}, N={N}, K={K}) ...", end=" ", flush=True)
        t0 = time.time()
        r = run_shape(shape_id, M, N, K, spec)
        t1 = time.time()
        print(f"done ({t1-t0:.1f}s)")
        results.append(r)

    print_main_table(results)
    print_speedup_table(results)
    print_legend()
    if args.detail:
        print_per_shape_detail(results)


if __name__ == "__main__":
    main()