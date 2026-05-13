"""
Main benchmarking entry point.

Usage:
    python benchmarks/bench_kernel.py --kernel triton --shape 1,4096,4096
    python benchmarks/bench_kernel.py --kernel cutlass --all-shapes
    python benchmarks/bench_kernel.py --compare-all

Outputs:
    - Console: TFLOPS, latency, bandwidth utilization
    - benchmarks/results/{kernel}_{shape}_{timestamp}.json
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from reference.torch_reference import (
    w8a8_scaled_mm_reference,
    quantize_per_token,
    quantize_per_channel,
)


# RTX 4090 hardware constants
RTX4090_INT8_TOPS_DENSE = 330e12    # 330 TOPS dense
RTX4090_HBM_BANDWIDTH = 1008e9      # 1008 GB/s


# LLaMA-7B test shapes
SHAPES = {
    "S1": (1, 4096, 4096),
    "S2": (16, 4096, 4096),
    "S3": (512, 4096, 4096),
    "S4": (2048, 4096, 4096),
}


def make_inputs(M, N, K, device="cuda", seed=42):
    """Generate reproducible test inputs."""
    torch.manual_seed(seed)
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=device)
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=device)
    sa = torch.rand(M, 1, dtype=torch.float32, device=device) * 0.1
    sb = torch.rand(1, N, dtype=torch.float32, device=device) * 0.01
    return a, b, sa, sb


def benchmark_kernel(kernel_fn, a, b, sa, sb, warmup=20, repeat=100):
    """
    Standard CUDA benchmark loop with torch.cuda.Event.
    
    Returns dict: {latency_us, tflops, bandwidth_gbps, bandwidth_util, compute_util}
    """
    M, K = a.shape
    _, N = b.shape
    
    # Warmup
    for _ in range(warmup):
        _ = kernel_fn(a, b, sa, sb)
    torch.cuda.synchronize()
    
    # Timed run
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        _ = kernel_fn(a, b, sa, sb)
    end.record()
    torch.cuda.synchronize()
    
    latency_ms = start.elapsed_time(end) / repeat
    latency_us = latency_ms * 1000
    
    # Theoretical: 2*M*N*K ops, M*K + K*N + M*N*2 bytes (int8 + int8 + fp16 out)
    flops = 2 * M * N * K
    bytes_moved = M * K + K * N + M * N * 2
    
    tflops = flops / (latency_ms * 1e-3) / 1e12
    bandwidth_gbps = bytes_moved / (latency_ms * 1e-3) / 1e9
    bandwidth_util = bandwidth_gbps / (RTX4090_HBM_BANDWIDTH / 1e9)
    compute_util = (tflops * 1e12) / RTX4090_INT8_TOPS_DENSE
    
    return {
        "latency_us": round(latency_us, 2),
        "tflops": round(tflops, 3),
        "bandwidth_gbps": round(bandwidth_gbps, 1),
        "bandwidth_util": round(bandwidth_util, 3),
        "compute_util": round(compute_util, 3),
        "shape": [M, N, K],
        "flops": flops,
        "bytes_moved": bytes_moved,
    }


def get_kernel(name: str):
    """Dispatch by name. Triton/CUTLASS implementations added in P2/P3."""
    if name == "reference":
        return w8a8_scaled_mm_reference
    elif name == "triton":
        # from triton_kernel.w8a8_mm import w8a8_scaled_mm_triton
        # return w8a8_scaled_mm_triton
        raise NotImplementedError("Triton kernel not yet implemented (P2)")
    elif name == "cutlass":
        raise NotImplementedError("CUTLASS kernel not yet implemented (P3)")
    else:
        raise ValueError(f"Unknown kernel: {name}")


def save_result(result: dict, kernel: str, shape_id: str):
    """Save benchmark result as JSON for tracking."""
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = out_dir / f"{kernel}_{shape_id}_{ts}.json"
    with open(filename, "w") as f:
        json.dump({
            "kernel": kernel,
            "shape_id": shape_id,
            "timestamp": ts,
            "gpu": torch.cuda.get_device_name(0),
            **result,
        }, f, indent=2)
    return filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["reference", "triton", "cutlass"], 
                        default="reference")
    parser.add_argument("--shape", type=str, help="M,N,K e.g. 1,4096,4096")
    parser.add_argument("--all-shapes", action="store_true", 
                        help="Run all LLaMA-7B shapes")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    kernel_fn = get_kernel(args.kernel)
    
    if args.all_shapes:
        shapes_to_run = list(SHAPES.items())
    elif args.shape:
        M, N, K = map(int, args.shape.split(","))
        shapes_to_run = [("custom", (M, N, K))]
    else:
        shapes_to_run = [("S2", SHAPES["S2"])]  # default
    
    print(f"=== Benchmarking {args.kernel} on {torch.cuda.get_device_name(0)} ===")
    print(f"{'Shape':<8} {'(M,N,K)':<20} {'Latency(us)':<12} "
          f"{'TFLOPS':<10} {'BW(GB/s)':<10} {'BW%':<8} {'Comp%':<8}")
    print("-" * 80)
    
    for shape_id, (M, N, K) in shapes_to_run:
        a, b, sa, sb = make_inputs(M, N, K)
        result = benchmark_kernel(kernel_fn, a, b, sa, sb)
        print(f"{shape_id:<8} {str((M,N,K)):<20} "
              f"{result['latency_us']:<12} "
              f"{result['tflops']:<10} "
              f"{result['bandwidth_gbps']:<10} "
              f"{result['bandwidth_util']*100:<7.1f}% "
              f"{result['compute_util']*100:<7.1f}%")
        if args.save:
            path = save_result(result, args.kernel, shape_id)
            print(f"  -> saved to {path}")


if __name__ == "__main__":
    main()
