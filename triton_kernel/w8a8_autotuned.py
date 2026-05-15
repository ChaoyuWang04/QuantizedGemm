"""
W8A8 scaled matmul - Triton kernel (v2: autotuned)

[2026-05-13 v2.1 update] 补充 BN=32 系列配置, 修复 S1/S2 上 autotune 选不到最优.

发现 (sweep_configs.py 扫描结果):
    S2 (M=16) 真正最优:  BM=16, BN=32, BK=128, w=4, s=3  -> 28.13us
    S1 (M=1)  真正最优:  BM=32, BN=32, BK=128, w=2, s=3  -> 28.74us
    
    之前候选列表里 BN 全是 64/128/256, 没有 32, 全军覆没.

原因:
    小 M 形状 (M=1, M=16) 只能在 N 维度上堆并行度.
    BN=32 时 grid_n = N/32, 给 GPU 提供 4x 更多 program 来填满 SM.
    BN=64/128 时 program 数太少, SM 闲置.

修复:
    候选列表加 6 个 BN=32 系列配置, 覆盖 S1/S2 真实最优.

预期 (替换前 vs 替换后, S2):
    替换前 autotune 选: BM=16, BN=64, BK=128, w=4, s=3 -> 30.85us (kernel only)
    替换后 autotune 选: BM=16, BN=32, BK=128, w=4, s=3 -> 28.13us (kernel only)
    节省约 2.7us, 相对 50us 总延迟约 -5%

设计哲学 (再次强调):
    "v1 -> v2 -> v2.1" 都是单变量改动:
        - v1: 无 autotune
        - v2:  加 autotune (候选 16 个)
        - v2.1: 候选扩充到 22 个, 多了小 M 友好配置
        Kernel 函数体一字未改, 精度保证 max_diff = 0.
"""

from typing import Optional
import torch
import triton
import triton.language as tl


# ============================================================
# Autotune 候选配置 (v2.1: 加 BN=32 系列覆盖小 M)
# ============================================================

_AUTOTUNE_CONFIGS = [
    # ============================================================
    # NEW (v2.1): 小 M 友好配置, BN=32 系列
    # 来源: tools/sweep_configs.py 扫描 144 配置后的 top 选择
    # ============================================================

    # S2 (M=16) 真实最优
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 128},
        num_warps=4, num_stages=3,
    ),  # 28.13us on S2, 28.93us on S1
    
    # S1 (M=1) 真实最优
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 128},
        num_warps=2, num_stages=3,
    ),  # 28.77us on S2, 28.74us on S1 - S1 best!

    # 备选: BK=256 系列, 减少 K-loop 迭代
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 256},
        num_warps=2, num_stages=3,
    ),  # 28.32us on S2, 28.99us on S1

    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 256},
        num_warps=2, num_stages=3,
    ),  # 29.25us on S2, 28.86us on S1

    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 256},
        num_warps=4, num_stages=3,
    ),  # 29.50us on S2, 28.83us on S1

    # 备选: BK=64, 不同 warps 组合
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 64},
        num_warps=4, num_stages=3,
    ),  # 29.76us on S2, 29.63us on S1

    # ============================================================
    # 原 v2 配置: Small M 场景 (M=1, decode batch=1)
    # ============================================================
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 64},
        num_warps=2, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=4, num_stages=3,
    ),

    # ============================================================
    # 原 v2 配置: Small M 场景 (M=16, batched decode)
    # ============================================================
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 128},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=4, num_stages=3,
    ),

    # ============================================================
    # 原 v2 配置: Medium M (M=64-256)
    # ============================================================
    triton.Config(
        {"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
        num_warps=4, num_stages=3,
    ),

    # ============================================================
    # 原 v2 配置: Large M (prefill, M=512+)
    # ============================================================
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},
        num_warps=8, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=8, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 64},
        num_warps=8, num_stages=3,
    ),

    # ============================================================
    # 原 v2 配置: Aggressive (试 shared mem 边缘)
    # ============================================================
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128},
        num_warps=8, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64},
        num_warps=8, num_stages=2,
    ),
]


# ============================================================
# Kernel: 函数体完全不动 (从 naive 到当前一字未改)
# ============================================================

@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["M", "N", "K"],
)
@triton.jit
def _w8a8_scaled_mm_kernel_autotuned(
    a_ptr, b_ptr, c_ptr,
    scale_a_ptr, scale_b_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_start in range(0, K, BLOCK_K):
        k_mask = offs_k[None, :] < (K - k_start)
        a_mask = k_mask & (offs_m[:, None] < M)
        b_mask = (offs_k[:, None] < (K - k_start)) & (offs_n[None, :] < N)

        a_tile = tl.load(a_ptrs, mask=a_mask, other=0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0)

        accumulator += tl.dot(a_tile, b_tile, out_dtype=tl.int32)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    sa_mask = offs_m < M
    sb_mask = offs_n < N
    sa = tl.load(scale_a_ptr + offs_m, mask=sa_mask, other=0.0)
    sb = tl.load(scale_b_ptr + offs_n, mask=sb_mask, other=0.0)

    c_fp32 = accumulator.to(tl.float32) * sa[:, None] * sb[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c_fp32.to(tl.float16), mask=c_mask)


# ============================================================
# Python wrapper (和之前一样, 已删除 squeeze)
# ============================================================

def w8a8_scaled_mm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Triton W8A8 scaled matmul (v2.1: autotuned with BN=32 configs)."""
    assert a.dtype == torch.int8, f"a must be int8, got {a.dtype}"
    assert b.dtype == torch.int8, f"b must be int8, got {b.dtype}"
    assert scale_a.dtype == torch.float32
    assert scale_b.dtype == torch.float32
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0]
    assert bias is None, "v2 暂不支持 bias"
    assert out_dtype == torch.float16, "v2 暂只支持 fp16 输出"

    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert scale_a.shape == (M, 1)
    assert scale_b.shape == (1, N)

    c = torch.empty((M, N), dtype=out_dtype, device=a.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _w8a8_scaled_mm_kernel_autotuned[grid](
        a, b, c,
        scale_a, scale_b,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return c


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from reference.torch_reference import (
        w8a8_scaled_mm_reference,
        quantize_per_token,
        quantize_per_channel,
    )

    print("=" * 70)
    print("Triton W8A8 v2.1 (more autotune configs) -- Smoke Test")
    print("=" * 70)

    torch.manual_seed(42)
    M, N, K = 16, 256, 256
    device = "cuda"

    x_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    print(f"\nNumber of autotune configs: {len(_AUTOTUNE_CONFIGS)}")
    print("First call (autotune will try all configs, may take 20-40s)...")
    import time
    t0 = time.time()
    out_triton = w8a8_scaled_mm_triton(x_q, w_q, sa, sb)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"First call: {t1-t0:.2f}s")

    print("\nSecond call (cached)...")
    t0 = time.time()
    out_triton = w8a8_scaled_mm_triton(x_q, w_q, sa, sb)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Second call: {(t1-t0)*1000:.3f}ms")

    out_ref = w8a8_scaled_mm_reference(x_q, w_q, sa, sb)
    max_diff = (out_triton.float() - out_ref.float()).abs().max().item()
    print(f"\nmax_abs_diff vs reference: {max_diff:.4e}")
    if max_diff < 1e-2:
        print("PASS (max_diff = 0 expected since algorithm unchanged)")
    else:
        print("FAIL — 算法没动, max_diff != 0, 检查 wrapper")