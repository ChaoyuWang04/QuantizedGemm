"""
Performance benchmark - W8A8 Triton kernel 多版本对比.

[2026-05-13] 加入 v3 swizzled 对比.

对手矩阵:
    1. Triton naive (v1)         - 第一版, 固定 BLOCK, baseline
    2. Triton autotuned (v2.1)   - autotune 选 BLOCK
    3. Triton swizzled (v3)      - autotune + GROUP_M swizzle
    4. cuBLAS fp16               - 替代方案对手

设计原则:
    - 时间 based warmup (0.5s 默认, 保证 GPU 充分热)
    - 全局 pre-warmup 5s
    - 100 次 repeat 取中位数

执行:
    python benchmarks/bench_kernel.py                    # 所有形状
    python benchmarks/bench_kernel.py --shape S2         # 只跑 S2
    python benchmarks/bench_kernel.py --warmup-s 1.0     # 更激进 warmup
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
from triton_kernel.w8a8_swizzled import w8a8_scaled_mm_triton as w8a8_swizzled


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
    # L40s: NVIDIA datasheet (Ada Lovelace, sm_89)
    # Note: int8_tops_dense 是 non-sparse 值; sparse 是 2x
    "NVIDIA L40S": {
        "int8_tops_dense": 362e12,
        "fp16_tflops": 181e12,
        "hbm_bandwidth": 864e9,
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
# 计时
# ============================================================

def benchmark_fn(fn, warmup_s: float = 0.5, repeat: int = 100) -> dict:
    """时间 based warmup 严格计时."""
    t_warmup_start = time.time()
    while time.time() - t_warmup_start < warmup_s:
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
# 全局 pre-warmup
# ============================================================

def global_prewarmup(duration_s: float = 5.0):
    """sustained workload 让 GPU 时钟进入 boost 状态."""
    print(f"Global GPU pre-warmup ({duration_s}s sustained workload)...")
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    y = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    t_start = time.time()
    n = 0
    while time.time() - t_start < duration_s:
        _ = torch.matmul(x, y)
        n += 1
    torch.cuda.synchronize()
    print(f"  -> ran {n} iterations, GPU should be at boost clock now")
    print()


# ============================================================
# 单形状测试 (4 个对手)
# ============================================================

def run_shape(shape_id: str, M: int, N: int, K: int, spec: dict,
              warmup_s: float = 0.5) -> dict:
    torch.manual_seed(42)

    x_fp16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    result = {"shape_id": shape_id, "M": M, "N": N, "K": K}

    # 1. Triton naive
    bench = benchmark_fn(lambda: w8a8_naive(x_q, w_q, sa, sb), warmup_s=warmup_s)
    result["triton_naive"] = {**bench, **compute_metrics(bench["latency_us"], M, N, K, spec)}

    # 2. Triton autotuned (v2.1)
    bench = benchmark_fn(lambda: w8a8_autotuned(x_q, w_q, sa, sb), warmup_s=warmup_s)
    result["triton_autotuned"] = {**bench, **compute_metrics(bench["latency_us"], M, N, K, spec)}

    # 3. Triton swizzled (v3) ★ NEW
    bench = benchmark_fn(lambda: w8a8_swizzled(x_q, w_q, sa, sb), warmup_s=warmup_s)
    result["triton_swizzled"] = {**bench, **compute_metrics(bench["latency_us"], M, N, K, spec)}

    # 4. cuBLAS fp16
    bench = benchmark_fn(lambda: cublas_fp16_baseline(x_fp16, w_fp16), warmup_s=warmup_s)
    result["cublas_fp16"] = {**bench, **compute_metrics(bench["latency_us"], M, N, K, spec)}

    # Speedup (基准: swizzled v3)
    base = result["triton_swizzled"]["latency_us"]
    result["speedup"] = {
        "naive_vs_swizzled": result["triton_naive"]["latency_us"] / base,
        "autotuned_vs_swizzled": result["triton_autotuned"]["latency_us"] / base,
        "swizzled_vs_cublas": result["cublas_fp16"]["latency_us"] / base,
    }

    return result


# ============================================================
# 打印
# ============================================================

def print_main_table(results: list):
    print()
    print("=" * 135)
    print("Performance Benchmark - W8A8 Triton (v1/v2.1/v3) vs cuBLAS fp16 (latency in us)")
    print("=" * 135)
    print()

    headers = ["Shape", "(M,N,K)", "naive", "autotuned", "swizzled", "cuBLAS",
               "v3 TFLOPS", "v3 Roof%", "Bound"]
    col_w = [6, 22, 10, 12, 10, 10, 12, 10, 10]

    line = ""
    for h, w in zip(headers, col_w):
        line += f"{h:<{w}}"
    print(line)
    print("-" * 135)

    for r in results:
        shape_str = f"({r['M']:>4}, {r['N']:>4}, {r['K']:>4})"
        row = [
            r["shape_id"], shape_str,
            f"{r['triton_naive']['latency_us']:.2f}",
            f"{r['triton_autotuned']['latency_us']:.2f}",
            f"{r['triton_swizzled']['latency_us']:.2f}",
            f"{r['cublas_fp16']['latency_us']:.2f}",
            f"{r['triton_swizzled']['tflops']:.2f}",
            f"{r['triton_swizzled']['roof_util']*100:.1f}%",
            r["triton_swizzled"]["bound"],
        ]
        line = ""
        for v, w in zip(row, col_w):
            line += f"{v:<{w}}"
        print(line)


def print_speedup_table(results: list):
    print()
    print("=" * 135)
    print("Speedup Analysis (基准: v3 swizzled, > 1x 表示 swizzled 更快)")
    print("=" * 135)
    print()

    headers = ["Shape", "naive -> swizzled", "autotuned -> swizzled (swizzle 收益)", "swizzled vs cuBLAS"]
    col_w = [6, 30, 40, 30]

    line = ""
    for h, w in zip(headers, col_w):
        line += f"{h:<{w}}"
    print(line)
    print("-" * 135)

    for r in results:
        sp = r["speedup"]
        x = sp["naive_vs_swizzled"]
        col2 = f"{x:.2f}x" + (" faster" if x > 1 else " slower")

        y = sp["autotuned_vs_swizzled"]
        if abs(y - 1.0) < 0.02:
            col3 = f"{y:.2f}x (essentially same)"
        else:
            col3 = f"{y:.2f}x" + (" faster (swizzle 有效)" if y > 1 else " slower (swizzle 失效)")

        z = sp["swizzled_vs_cublas"]
        col4 = f"{z:.2f}x" + (" faster" if z > 1 else " slower")

        line = f"{r['shape_id']:<{col_w[0]}}{col2:<{col_w[1]}}{col3:<{col_w[2]}}{col4:<{col_w[3]}}"
        print(line)


def print_legend():
    print()
    print("=" * 135)
    print("解读")
    print("=" * 135)
    print()
    print("对手身份:")
    print("  naive (v1)     = 固定 BLOCK=64, baseline")
    print("  autotuned (v2.1) = autotune 选 BLOCK 配置")
    print("  swizzled (v3)  = autotune + GROUP_M swizzle 提升 L2 命中")
    print("  cuBLAS fp16    = 替代方案")
    print()
    print("怎么读 swizzle 收益:")
    print("  autotuned -> swizzled > 1.0x: swizzle 有效 (主要看 S3/S4)")
    print("  autotuned -> swizzled ≈ 1.0x: swizzle 无效 (小形状常见, autotune 选 GROUP_M=1)")
    print("  autotuned -> swizzled < 1.0x: swizzle 反效果 (不应出现, 因为 GROUP_M=1 是 fallback)")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=list(SHAPES.keys()) + ["all"], default="all")
    parser.add_argument("--warmup-s", type=float, default=0.5)
    parser.add_argument("--no-prewarmup", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    spec = HARDWARE_SPECS.get(gpu_name, DEFAULT_SPEC)
    print(f"GPU:                 {gpu_name}")
    print(f"INT8 TOPS (dense):   {spec['int8_tops_dense']/1e12:.0f}")
    print(f"HBM Bandwidth:       {spec['hbm_bandwidth']/1e9:.0f} GB/s")
    print(f"Warmup per bench:    {args.warmup_s:.1f}s")
    if gpu_name not in HARDWARE_SPECS:
        print(f"WARN: GPU not in spec table, using 4090 fallback")
    print()

    if not args.no_prewarmup:
        global_prewarmup(duration_s=5.0)

    if args.shape == "all":
        shapes_to_run = list(SHAPES.items())
    else:
        shapes_to_run = [(args.shape, SHAPES[args.shape])]

    print("Note: 第一次跑 autotuned/swizzled 会触发 autotune (每形状 20-60s)")
    print()

    results = []
    for shape_id, (M, N, K) in shapes_to_run:
        print(f"Running {shape_id} (M={M}, N={N}, K={K}) ...", end=" ", flush=True)
        t0 = time.time()
        r = run_shape(shape_id, M, N, K, spec, warmup_s=args.warmup_s)
        t1 = time.time()
        print(f"done ({t1-t0:.1f}s)")
        results.append(r)

    print_main_table(results)
    print_speedup_table(results)
    print_legend()


if __name__ == "__main__":
    main()