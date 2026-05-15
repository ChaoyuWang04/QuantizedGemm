"""
W8A8 scaled matmul - Triton kernel (v2: autotuned)

相对 v1 (naive) 的唯一变化:
    加上 @triton.autotune, 让编译器自动从候选配置里选最快的.
    Kernel 内部逻辑完全没变 (这样精度保证还是 max_diff = 0).

设计哲学:
    "v1 -> v2" 是一次 单变量优化:
        - v1 算法对了 (max_diff = 0)
        - v2 只动调参 (BLOCK 大小), 算法不变
        - 如果 v2 出现 max_diff != 0, 一定是 autotune 配置有 bug, 不是算法 bug
        这种 "一次只动一件事" 的纪律是 kernel 优化的铁律.

设计目标:
    1. 在 S1-S4 四个形状上, 每个都用上"差不多最优"的 BLOCK 配置
    2. autotune 第一次跑要 < 30 秒 (候选不能太多)
    3. 缓存命中后, 性能应该 >= v1 (理论上, autotune 总能找到 >= 固定配置的)

预期收益 (相对 v1):
    S1 (M=1):     +20-50%  (autotune 选小 BLOCK_M, 减少浪费)
    S2 (M=16):    +200-500% (这是 v1 的死穴: 固定 BLOCK_M=64, 75% 算力浪费)
    S3 (M=512):   +100-200% (autotune 选大 BLOCK, Tensor Core 占用率上去)
    S4 (M=2048):  +100-200% (同上)

不在 v2 实现的 (留给 v3+):
    - swizzle / super-grouping (改善 L2 命中)
    - split-K (M 很小时 K 维度并行)
    - bias / multi-dtype 输出
"""

from typing import Optional
import torch
import triton
import triton.language as tl


# ============================================================
# Autotune 候选配置
# ============================================================
# 这是 v2 的核心. 设计原则:
#   1. 必须满足硬件约束:
#      - BLOCK_M, BLOCK_N >= 16 (mma 最小形状)
#      - BLOCK_K >= 32 (int8 mma 要求)
#      - BLOCK_M/N 必须能被 16 整除
#   2. 覆盖三大场景: small M (decode), medium M (small prefill), large M (prefill)
#   3. shared memory 不超限 (Ampere ~100KB):
#      A tile + B tile + double/triple buffer
#      = num_stages * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) bytes
#   4. 总数控制在 15-20 个: 多了 autotune 太慢, 少了可能漏最优
#
# Triton 把 5090 (sm_120) 当 sm_80 (Ampere) 用, 所以这套配置基本和 A100 一样.

_AUTOTUNE_CONFIGS = [
    # ----- Small M 场景: M=1 (decode batch=1) -----
    # 关键: BLOCK_M 小, BLOCK_N 可以大 (N 维度可以充分并行)
    # num_warps 不需要大, M 维度本来就没多少 work
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

    # ----- Small M 场景: M=16 (decode batch=16, v1 的死穴) -----
    # 关键: BLOCK_M = M, 不浪费; BLOCK_N 大, BLOCK_K 大, 让 K-loop 短
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
    # 平衡型配置, BLOCK_M/N 接近 1:1 或 1:2
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
    # 关键: BLOCK_M, BLOCK_N 都大, 充分利用 Tensor Core
    # num_warps=8 让一个 program 用更多并行线程
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

    # ----- Aggressive 配置 (可能编译失败, autotune 会自动跳过) -----
    # 这些试试 shared mem 边缘, 万一性能炸裂呢
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
# Triton kernel: 加 @triton.autotune 装饰器
# ============================================================
# 重点: kernel 函数体和 v1 一字不差, 只多了一个装饰器.
# 这是优化的纪律: 单变量改动, 便于追溯 bug.

@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["M", "N", "K"],   # 形状变了才重新 autotune; 同形状用缓存
)
@triton.jit
def _w8a8_scaled_mm_kernel_autotuned(
    # ----- 数据指针 -----
    a_ptr, b_ptr, c_ptr,
    scale_a_ptr, scale_b_ptr,
    # ----- 矩阵维度 -----
    M, N, K,
    # ----- Stride -----
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # ----- Block size (autotune 自动填充, 不再由调用方指定) -----
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    与 v1 函数体完全相同. 唯一区别:
        1. BLOCK_M/N/K 由 @triton.autotune 自动决定, 不再是调用方传入
        2. num_warps / num_stages 也由 autotune 自动选 (隐式传入)
    """
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
# Python wrapper: 不再传 BLOCK_M/N/K, autotune 自动决定
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
    Triton W8A8 scaled matmul (v2 autotuned).

    与 v1 (naive) 的对外接口区别:
        - 不再接受 BLOCK_M/N/K 参数 (autotune 自动选)
        - 其他完全一样

    第一次调用某个形状会卡几秒 (autotune 在试候选配置),
    之后同形状直接用缓存, 和 v1 一样快.
    """
    # ----- 输入校验 (和 v1 完全一样) -----
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

    sa_1d = scale_a.squeeze(-1).contiguous()
    sb_1d = scale_b.squeeze(0).contiguous()

    # ----- 关键变化: grid 用 lambda, BLOCK 是动态的 -----
    # v1: grid = (cdiv(M, 64), cdiv(N, 64))    # 固定 BLOCK
    # v2: grid 不能在外面写死, 因为 autotune 会用不同的 BLOCK_M/N
    #     Triton 的解决方案: grid 写成 lambda, 接受 meta 参数,
    #     meta 里包含 autotune 选出的 BLOCK_M/N
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    # 调用时不再传 BLOCK_M/N/K, autotune 会从候选里选并填进 meta
    _w8a8_scaled_mm_kernel_autotuned[grid](
        a, b, c,
        sa_1d, sb_1d,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return c


# ============================================================
# Smoke test: 同 v1 完全一样, 应该 max_diff = 0
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
    print("Triton W8A8 v2 (autotuned) -- Smoke Test")
    print("=" * 70)

    torch.manual_seed(42)
    M, N, K = 16, 256, 256
    device = "cuda"

    x_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    print("\nFirst call (will trigger autotune, may take a few seconds)...")
    import time
    t0 = time.time()
    out_triton = w8a8_scaled_mm_triton(x_q, w_q, sa, sb)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"First call: {t1-t0:.2f}s")

    print("\nSecond call (cached, should be fast)...")
    t0 = time.time()
    out_triton = w8a8_scaled_mm_triton(x_q, w_q, sa, sb)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Second call: {(t1-t0)*1000:.3f}ms")

    out_ref = w8a8_scaled_mm_reference(x_q, w_q, sa, sb)

    max_diff = (out_triton.float() - out_ref.float()).abs().max().item()
    print(f"\nmax_abs_diff vs reference: {max_diff:.4e}")
    if max_diff < 1e-2:
        print(f"PASS (< 0.01 target)")
    else:
        print(f"FAIL (>= 0.01 target)")