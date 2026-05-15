"""
Performance benchmark - W8A8 Triton kernel 多版本对比.

[2026-05-13 修复] Warmup 改为时间 based, 加全局 GPU pre-warmup.

为什么改:
    旧版用 "warmup=10 次" 固定次数. 但 kernel 跑得快 (28us) 时,
    10 次只有 0.3ms, GPU 时钟还没 boost (NVIDIA GPU 需要 sustained
    workload 0.5-2s 才进入高时钟状态).
    
    实测 stability_diagnosis.py 用 1s warmup 给出 naive S2 = 48us,
    bench_kernel.py 用 10 次 warmup 给出 naive S2 = 89us. 差 40us
    全部是冷 GPU 拖累.

修复:
    1. benchmark_fn 改为时间 based warmup (默认 0.5s)
    2. main() 开头先全局 pre-warmup 5s (跑任意 workload 让 GPU 热起来)

对手:
    1. Triton W8A8 naive (v1)
    2. Triton W8A8 autotuned (v2.1)
    3. cuBLAS fp16

执行:
    python benchmarks/bench_kernel.py                    # 所有形状
    python benchmarks/bench_kernel.py --shape S2         # 只跑 S2
    python benchmarks/bench_kernel.py --warmup-s 2.0     # 更激进 warmup
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
# 修复后的 benchmark_fn: 时间 based warmup
# ============================================================

def benchmark_fn(fn, warmup_s: float = 0.5, repeat: int = 100) -> dict:
    """
    严格计时, warmup 用时间 (不是次数).
    
    Args:
        fn: 要测的 kernel 调用函数 (lambda)
        warmup_s: warmup 持续时间, 默认 0.5s. 慢 kernel 跑次数少,
                  快 kernel 跑次数多, 但都保证 GPU 热到稳定时钟.
        repeat: 正式测量次数, 100 次取中位数
    """
    # ----- 时间 based warmup -----
    t_warmup_start = time.time()
    n_warmup = 0
    while time.time() - t_warmup_start < warmup_s:
        fn()
        n_warmup += 1
    torch.cuda.synchronize()

    # ----- 正式测量 -----
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
        "n_warmup": n_warmup,
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
# 全局 pre-warmup: 让 GPU 时钟先稳定
# ============================================================

def global_prewarmup(duration_s: float = 5.0):
    """
    在 benchmark 任何形状前, 用一个 sustained workload 让 GPU 进入
    稳定高时钟状态. 这避免"第一个测试形状"撞冷启动.
    
    用一个中等大小的 cuBLAS matmul 当 prewarmup 负载 (确定能跑且占满 GPU).
    """
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
# 单形状测试
# ============================================================

def run_shape(shape_id: str, M: int, N: int, K: int, spec: dict,
              warmup_s: float = 0.5) -> dict:
    torch.manual_seed(42)

    x_fp16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    result = {"shape_id": shape_id, "M": M, "N": N, "K": K}

    # 1. Triton naive (v1)
    bench = benchmark_fn(lambda: w8a8_naive(x_q, w_q, sa, sb), warmup_s=warmup_s)
    result["triton_naive"] = {**bench, **compute_metrics(bench["latency_us"], M, N, K, spec)}

    # 2. Triton autotuned (v2.1)
    bench = benchmark_fn(lambda: w8a8_autotuned(x_q, w_q, sa, sb), warmup_s=warmup_s)
    result["triton_autotuned"] = {**bench, **compute_metrics(bench["latency_us"], M, N, K, spec)}

    # 3. cuBLAS fp16
    bench = benchmark_fn(lambda: cublas_fp16_baseline(x_fp16, w_fp16), warmup_s=warmup_s)
    result["cublas_fp16"] = {**bench, **compute_metrics(bench["latency_us"], M, N, K, spec)}

    # Speedup
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
    print("=" * 120)
    print("Performance Benchmark - W8A8 Triton vs cuBLAS fp16 (latency in microseconds)")
    print("=" * 120)
    print()

    headers = ["Shape", "(M,N,K)", "naive", "autotuned", "cuBLAS fp16",
               "autotuned TFLOPS", "Roof%", "Bound"]
    col_w = [6, 22, 11, 12, 13, 18, 8, 10]

    line = ""
    for h, w in zip(headers, col_w):
        line += f"{h:<{w}}"
    print(line)
    print("-" * 120)

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
    print("=" * 120)
    print("Speedup Analysis (基准: Triton autotuned, 数字>1 表示 autotuned 更快)")
    print("=" * 120)
    print()

    headers = ["Shape", "naive -> autotuned (优化进度)", "autotuned vs cuBLAS fp16"]
    col_w = [6, 40, 40]

    line = ""
    for h, w in zip(headers, col_w):
        line += f"{h:<{w}}"
    print(line)
    print("-" * 120)

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
    print("=" * 120)
    print("解读")
    print("=" * 120)
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


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=list(SHAPES.keys()) + ["all"], default="all")
    parser.add_argument("--warmup-s", type=float, default=0.5,
                        help="每个 benchmark 的 warmup 时间, 默认 0.5s")
    parser.add_argument("--no-prewarmup", action="store_true",
                        help="跳过全局 GPU pre-warmup")
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
    print(f"Warmup per bench:    {args.warmup_s:.1f}s")
    if gpu_name not in HARDWARE_SPECS:
        print(f"WARN: GPU not in spec table, using 4090 fallback")
    print()

    # 全局 pre-warmup 让 GPU 时钟稳定
    if not args.no_prewarmup:
        global_prewarmup(duration_s=5.0)

    if args.shape == "all":
        shapes_to_run = list(SHAPES.items())
    else:
        shapes_to_run = [(args.shape, SHAPES[args.shape])]

    print("Note: 第一次跑 autotuned 版本会触发 autotune (每形状 20-40s)")
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