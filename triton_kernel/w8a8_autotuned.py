"""
W8A8 scaled matmul - Triton kernel (v2: autotuned)

[2026-05-13 更新] 移除 wrapper 中冗余的 squeeze().contiguous().

发现:
    profile_wrapper.py 显示, S2 (M=16) 上 wrapper overhead 约 32us, 而
    kernel 本身只要 30us. 32us 里有 8-12us 来自两个 squeeze().contiguous() 调用.

原因:
    scale_a 形状 [M, 1] 在内存里就是 M 个连续 fp32, 和 [M] 的 1D 张量
    完全等价. kernel 内部用裸指针 + offset 访问, 根本不在乎传 [M, 1] 还是 [M].
    squeeze() 创建了完全多余的视图, contiguous() 还可能触发不必要的拷贝.

修复:
    wrapper 不再 squeeze, 直接传 [M, 1] 给 kernel.
    Kernel 内部的 tl.load(scale_a_ptr + offs_m) 行为完全不变.

预期收益:
    S2 wrapper 总延迟: 63us -> 约 50-55us (-15-20%)
    S1 同样改善
    S3/S4 改善较小 (overhead 占比小)

设计哲学 (再次强调):
    "v1 -> v2 -> v2 优化版" 都是单变量改动:
        - v1 -> v2:  加 autotune (改了配置选择)
        - v2 -> 当前: 删 squeeze (改了 wrapper, 不动 kernel)
        Kernel 内部三个版本完全相同, 精度保证还是 max_diff = 0.
"""

from typing import Optional
import torch
import triton
import triton.language as tl


# ============================================================
# Autotune 候选配置 (和之前完全一样)
# ============================================================

_AUTOTUNE_CONFIGS = [
    # ----- Small M 场景: M=1 (decode batch=1) -----
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 64},
        num_warps=2, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64},
        num_warps=4, num_stages=2,
    ),

    # ----- Small M 场景: M=16 -----
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 128},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128},
        num_warps=4, num_stages=2,
    ),

    # ----- Medium M 场景: M=64~256 -----
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

    # ----- Large M 场景: M=512+ (prefill) -----
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

    # ----- Aggressive 配置 -----
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

    # 加载 scale.
    # 关键观察: scale_a 在 wrapper 里传进来时形状是 [M, 1], 但它在内存里
    # 就是 M 个连续 fp32, 和 [M] 完全一样. tl.load(scale_a_ptr + offs_m)
    # 用裸指针 + 1D offset 访问, 根本不在乎调用方的 .shape 是什么.
    # 之前 wrapper 里的 squeeze().contiguous() 完全多余, 已在新 wrapper 中删除.
    sa_mask = offs_m < M
    sb_mask = offs_n < N
    sa = tl.load(scale_a_ptr + offs_m, mask=sa_mask, other=0.0)
    sb = tl.load(scale_b_ptr + offs_n, mask=sb_mask, other=0.0)

    c_fp32 = accumulator.to(tl.float32) * sa[:, None] * sb[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c_fp32.to(tl.float16), mask=c_mask)


# ============================================================
# Python wrapper: 精简版, 删除 squeeze().contiguous()
# ============================================================

def w8a8_scaled_mm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Triton W8A8 scaled matmul (v2 autotuned, wrapper 优化版).

    相对前一版 wrapper 的唯一变化:
        - 删除 scale_a.squeeze(-1).contiguous() 和 scale_b.squeeze(0).contiguous()
        - 直接把 [M, 1] 和 [1, N] 形状的 scale 传给 kernel
        - kernel 内部行为完全不变

    精度保证: 和前一版一致 (max_diff = 0)
    """
    # ----- 输入校验 (保留, 但是 v1/v2 都是 8 个 assert, 这部分耗时是固有成本) -----
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

    # 关键变化: 不再调用 squeeze().contiguous()
    # scale_a [M, 1] 在内存里就是 M 个连续 fp32, kernel 内用裸指针访问即可.
    # 直接把 scale tensor 传过去, kernel 内的 tl.load(scale_a_ptr + offs_m)
    # 会正确访问到第 offs_m 个 fp32 元素.

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _w8a8_scaled_mm_kernel_autotuned[grid](
        a, b, c,
        scale_a, scale_b,   # 直接传 2D tensor, 不 squeeze
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
    print("Triton W8A8 v2 (autotuned, wrapper optimized) -- Smoke Test")
    print("=" * 70)

    torch.manual_seed(42)
    M, N, K = 16, 256, 256
    device = "cuda"

    x_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    print("\nFirst call (will trigger autotune)...")
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
        print("FAIL — 算法应该没动, 但 max_diff != 0, 检查 wrapper 改动")