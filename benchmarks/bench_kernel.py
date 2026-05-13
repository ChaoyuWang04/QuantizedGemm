"""
W8A8 GEMM 单算子 benchmark 入口。

这个脚本负责构造固定 shape 的 int8 输入,选择 reference/Triton/CUTLASS kernel,
用 CUDA Event 统计延迟,并计算 TFLOPS、带宽利用率和算力利用率。

用法:
    python benchmarks/bench_kernel.py --kernel reference --shape 1,4096,4096
    python benchmarks/bench_kernel.py --kernel triton --all-shapes

输出:
    - 终端表格:latency、TFLOPS、带宽和算力利用率。
    - 可选 JSON:benchmarks/results/{kernel}_{shape}_{timestamp}.json。
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

import sys

# 允许从仓库根目录导入 reference 模块,方便直接运行本脚本。
sys.path.insert(0, str(Path(__file__).parent.parent))
from reference.torch_reference import (
    w8a8_scaled_mm_reference,
    quantize_per_token,
    quantize_per_channel,
)


# RTX 4090 硬件常数,用于把实测速度换算成利用率。
RTX4090_INT8_TOPS_DENSE = 330e12    # 330 TOPS dense
RTX4090_HBM_BANDWIDTH = 1008e9      # 1008 GB/s


# LLaMA-7B 常见投影层 shape,覆盖 decode 和 prefill 场景。
SHAPES = {
    "S1": (1, 4096, 4096),
    "S2": (16, 4096, 4096),
    "S3": (512, 4096, 4096),
    "S4": (2048, 4096, 4096),
}


def make_inputs(M, N, K, device="cuda", seed=42):
    """
    生成可复现的随机 int8 输入和 fp32 scale。

    返回:
        a/b 是 int8 矩阵,sa/sb 分别是 per-token 和 per-channel scale。
    """
    torch.manual_seed(seed)
    # 激活和权重直接采样 int8,模拟已经量化后的输入。
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=device)
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=device)
    # scale 范围参考需求文档中的真实分布假设。
    sa = torch.rand(M, 1, dtype=torch.float32, device=device) * 0.1
    sb = torch.rand(1, N, dtype=torch.float32, device=device) * 0.01
    return a, b, sa, sb


def benchmark_kernel(kernel_fn, a, b, sa, sb, warmup=20, repeat=100):
    """
    用 CUDA Event 对单个 kernel 做标准 benchmark。

    warmup 用于排除首次运行开销,repeat 用于平均多次运行的延迟。

    返回:
        包含 latency_us、tflops、bandwidth_gbps、bandwidth_util、compute_util 的字典。
    """
    M, K = a.shape
    _, N = b.shape

    # 预热 kernel,让 CUDA context、缓存和 JIT 等开销不进入计时。
    for _ in range(warmup):
        _ = kernel_fn(a, b, sa, sb)
    torch.cuda.synchronize()

    # 使用 CUDA Event 在 GPU 时间线上计时,比 CPU time 更准确。
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        _ = kernel_fn(a, b, sa, sb)
    end.record()
    torch.cuda.synchronize()
    
    latency_ms = start.elapsed_time(end) / repeat
    latency_us = latency_ms * 1000

    # 理论计算量:矩阵乘是 2*M*N*K ops。
    flops = 2 * M * N * K
    # 粗略访存量:int8 A + int8 B + fp16 output,忽略 scale 和缓存复用。
    bytes_moved = M * K + K * N + M * N * 2

    # 将平均延迟换算为吞吐、带宽和相对 4090 峰值的利用率。
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
    """
    根据命令行传入的名字选择 kernel 函数。

    Triton 和 CUTLASS 还未实现时显式报错,避免 benchmark 跑到空实现。
    """
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
    """
    将 benchmark 结果保存成 JSON,用于后续性能回归和画图。

    文件名包含 kernel、shape 和时间戳,避免覆盖历史结果。
    """
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = out_dir / f"{kernel}_{shape_id}_{ts}.json"
    with open(filename, "w") as f:
        # 写入硬件名、shape、吞吐等完整上下文,方便离线分析。
        json.dump({
            "kernel": kernel,
            "shape_id": shape_id,
            "timestamp": ts,
            "gpu": torch.cuda.get_device_name(0),
            **result,
        }, f, indent=2)
    return filename


def main():
    """解析命令行参数,选择 shape/kernel,执行 benchmark 并打印表格。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["reference", "triton", "cutlass"], 
                        default="reference")
    parser.add_argument("--shape", type=str, help="M,N,K e.g. 1,4096,4096")
    parser.add_argument("--all-shapes", action="store_true", 
                        help="Run all LLaMA-7B shapes")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    # 当前 benchmark 依赖 CUDA Event,没有 GPU 时直接退出。
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    kernel_fn = get_kernel(args.kernel)

    # 命令行可以指定所有标准 shape、单个自定义 shape,或使用默认 S2。
    if args.all_shapes:
        shapes_to_run = list(SHAPES.items())
    elif args.shape:
        M, N, K = map(int, args.shape.split(","))
        shapes_to_run = [("custom", (M, N, K))]
    else:
        shapes_to_run = [("S2", SHAPES["S2"])]  # default

    # 打印固定宽度表头,方便对比不同 shape 的性能。
    print(f"=== Benchmarking {args.kernel} on {torch.cuda.get_device_name(0)} ===")
    print(f"{'Shape':<8} {'(M,N,K)':<20} {'Latency(us)':<12} "
          f"{'TFLOPS':<10} {'BW(GB/s)':<10} {'BW%':<8} {'Comp%':<8}")
    print("-" * 80)

    for shape_id, (M, N, K) in shapes_to_run:
        # 每个 shape 独立生成输入,保证结果可复现。
        a, b, sa, sb = make_inputs(M, N, K)
        result = benchmark_kernel(kernel_fn, a, b, sa, sb)
        # 控制台输出只保留关键性能指标,详细上下文可选写 JSON。
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
