"""
大形状 smoke test - 在 LLaMA-7B 4 个形状上验证 Triton kernel 精度。

跑这个脚本的目的:
    1. 小形状 (16, 256, 256) 已经验证 Triton 输出 = reference 精确等价
    2. 大形状下 int32 累加值远大, 可能暴露:
       - int32 溢出 (理论 K>131072 才会, 但 K=4096 极端值仍要验证)
       - fp16 输出溢出 (累加 × scale 接近 fp16 上限)
       - Tensor Core 对齐边界 (M=1 这种极端形状)
    3. 输出对比表, 给你直观的"安全裕度"感受

非 pytest 性质 - 这是"看一眼"的 smoke test, 给出多形状综合视图。

执行:
    python tests/smoke_large_shapes.py

期望 (假设 v1 实现正确):
    所有 S1-S4 形状下, max_abs_diff < 1e-2 (P0 目标)
    最好情况 max_diff = 0 (小形状情况)
    实际情况 max_diff < 1e-3 量级 (大形状由于 fp32 ULP 可能有微小差异)
"""

import sys
from pathlib import Path
from typing import Optional

import torch

# 让 reference 和 triton 模块可被 import
sys.path.insert(0, str(Path(__file__).parent.parent))
from reference.torch_reference import (
    w8a8_scaled_mm_reference,
    quantize_per_token,
    quantize_per_channel,
)
from triton_kernel.w8a8_mm import w8a8_scaled_mm_triton


# LLaMA-7B 测试形状
SHAPES = [
    ("S1", 1, 4096, 4096, "decode batch=1, 极端 memory-bound"),
    ("S2", 16, 4096, 4096, "decode batch=16, memory-bound"),
    ("S3", 512, 4096, 4096, "prefill seq=512, balanced"),
    ("S4", 2048, 4096, 4096, "prefill seq=2048, compute-bound"),
]

# 精度目标 (来自 P0 文档)
TARGET_MAX_DIFF = 1e-2

# 表格分隔线
SEP = "=" * 100


def run_one_shape(shape_id: str, M: int, N: int, K: int, scenario: str) -> dict:
    """跑一个形状的精度对比, 返回结果字典."""
    torch.manual_seed(42)

    # ---- 构造真实分布的输入 ----
    # 用 randn 模拟真实激活/权重分布, weight 乘 0.1 让 scale 更接近模型
    x_fp16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1

    # 量化
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    # ---- 跑 Triton 和 reference, 对比 ----
    out_triton = w8a8_scaled_mm_triton(x_q, w_q, sa, sb)
    out_ref = w8a8_scaled_mm_reference(x_q, w_q, sa, sb)

    # ---- 各种统计 ----
    diff_abs = (out_triton.float() - out_ref.float()).abs()
    max_diff = diff_abs.max().item()
    mean_diff = diff_abs.mean().item()
    absmax_out = out_ref.abs().max().item()
    rel_diff = max_diff / absmax_out if absmax_out > 0 else 0.0

    # ---- 健康检查 ----
    has_nan = torch.isnan(out_triton).any().item()
    has_inf = torch.isinf(out_triton).any().item()

    # ---- 安全裕度 ----
    safety_margin = TARGET_MAX_DIFF / max(max_diff, 1e-12)

    return {
        "shape_id": shape_id,
        "M": M, "N": N, "K": K,
        "scenario": scenario,
        "out_absmax": absmax_out,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rel_diff": rel_diff,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "safety_margin": safety_margin,
        "passed": (max_diff < TARGET_MAX_DIFF) and (not has_nan) and (not has_inf),
    }


def print_results_table(results: list[dict]):
    """打印对齐的对比表."""
    print(SEP)
    print("Triton W8A8 v1  vs  PyTorch Reference  -- 大形状精度对比")
    print(SEP)
    print()

    # 表头
    print(f"{'Shape':<6} {'(M, N, K)':<22} {'absmax':<10} "
          f"{'max_diff':<14} {'mean_diff':<14} {'rel_diff':<10} "
          f"{'safety':<12} {'status':<10}")
    print("-" * 100)

    # 各行数据
    for r in results:
        shape_str = f"({r['M']:>4}, {r['N']:>4}, {r['K']:>4})"
        status = "PASS" if r["passed"] else "FAIL"
        if r["has_nan"]:
            status = "NaN"
        elif r["has_inf"]:
            status = "Inf"

        # 安全裕度: target / max_diff, 越大越好
        if r["max_diff"] < 1e-12:
            safety_str = "perfect"   # max_diff = 0 时, 无限裕度
        elif r["safety_margin"] > 1000:
            safety_str = f"{r['safety_margin']:.0f}x"
        else:
            safety_str = f"{r['safety_margin']:.1f}x"

        print(f"{r['shape_id']:<6} {shape_str:<22} "
              f"{r['out_absmax']:<10.4f} "
              f"{r['max_diff']:<14.4e} "
              f"{r['mean_diff']:<14.4e} "
              f"{r['rel_diff']*100:<9.4f}% "
              f"{safety_str:<12} {status:<10}")

    print()
    print(f"Target: max_diff < {TARGET_MAX_DIFF} (P0 目标)")
    print(f"safety = (target / max_diff), 表示离阈值还有多远")


def print_scenario_details(results: list[dict]):
    """对每个形状打印额外的场景说明."""
    print()
    print(SEP)
    print("场景注释")
    print(SEP)
    for r in results:
        status_icon = "[OK]" if r["passed"] else "[!!]"
        print(f"  {status_icon} {r['shape_id']} {r['scenario']}")


def print_summary(results: list[dict]):
    """汇总判断."""
    print()
    print(SEP)
    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)

    if n_pass == n_total:
        print(f"All {n_total}/{n_total} shapes passed precision target.")
        print()
        # 看最差的形状裕度
        worst = max(results, key=lambda r: r["max_diff"])
        if worst["max_diff"] < 1e-12:
            print("Best case: All shapes have max_diff = 0 (bit-exact equivalence).")
        else:
            print(f"Worst case: {worst['shape_id']} with max_diff = {worst['max_diff']:.4e}, "
                  f"safety margin = {worst['safety_margin']:.1f}x")
            print("This is healthy. Triton kernel can proceed to performance optimization.")
    else:
        print(f"FAILED: only {n_pass}/{n_total} shapes passed.")
        print("Do NOT proceed to performance optimization until all shapes pass.")
        for r in results:
            if not r["passed"]:
                print(f"  Failed shape: {r['shape_id']} (M={r['M']}, N={r['N']}, K={r['K']}), "
                      f"max_diff = {r['max_diff']:.4e}")
    print(SEP)


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton kernel: w8a8_scaled_mm_triton (v1 naive)")
    print(f"Reference:     w8a8_scaled_mm_reference (CPU int32 matmul)")
    print()

    results = []
    for shape_id, M, N, K, scenario in SHAPES:
        print(f"Running {shape_id} (M={M}, N={N}, K={K}) ...", end=" ", flush=True)
        try:
            r = run_one_shape(shape_id, M, N, K, scenario)
            print("done")
            results.append(r)
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                "shape_id": shape_id, "M": M, "N": N, "K": K,
                "scenario": scenario, "passed": False, "error": str(e),
                "out_absmax": 0, "max_diff": float('inf'), "mean_diff": 0,
                "rel_diff": 0, "has_nan": False, "has_inf": False,
                "safety_margin": 0,
            })

    print_results_table(results)
    print_scenario_details(results)
    print_summary(results)


if __name__ == "__main__":
    main()