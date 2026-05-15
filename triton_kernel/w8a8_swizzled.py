"""
W8A8 scaled matmul - Triton kernel (v3: swizzled)

[2026-05-13 v3] 加 GROUP_M swizzle, 提升大形状的 L2 cache 命中率.

核心思路:
    默认 Triton row-major 启动顺序下, 同时跑的 ~128 个 program 在 N 维度上
    跨度大 (读 32 段不同的 B), 导致 L2 cache 上 B 压力大.
    
    GROUP_M swizzle 让同时跑的 program 在 2D 上聚集成 GROUP_M 行 × N 列的
    小矩形, B 在 L2 上的压力减半 (16 段 vs 32 段).

适用场景:
    - 大形状 (M >= 512), L2 复用空间大 -> 显著收益
    - 小形状 (M < 256), L2 反正够用 -> 收益接近 0
    
    所以 v3 应该:
    - 大 M 配置加 swizzle 候选 (GROUP_M=4, 8, 16)
    - 小 M 配置只保留 GROUP_M=1 (等价于关 swizzle)

GROUP_M=1 = 关 swizzle:
    swizzle 公式在 GROUP_M=1 时退化为 row-major, 行为 = v2.1.
    所以 GROUP_M=1 是 safety net, autotune 选它就等于退回 v2.1.

预期收益 (vs v2.1):
    S1, S2 (小 M, memory-bound): 持平 (swizzle 无效, 退化到 GROUP_M=1)
    S3 (M=512, balanced): +5-15%
    S4 (M=2048, compute-bound): +20-30%, Roof% 从 43% 推到 55-65%

不变的事:
    - Kernel 数学逻辑没变 (mma 顺序、累加、dequant 全和 v2.1 一样)
    - 精度仍然 max_diff = 0
"""

from typing import Optional
import torch
import triton
import triton.language as tl


# ============================================================
# Autotune 候选: 大 M 加 swizzle 候选, 小 M 用 GROUP_M=1 退化
# ============================================================

_AUTOTUNE_CONFIGS = [
    # ============================================================
    # Small M 配置 (M=1, M=16): GROUP_M=1 退化为 row-major
    # 这些配置来自 v2.1 的 sweep 结果, swizzle 在小形状无收益
    # ============================================================
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 128, "GROUP_M": 1},
        num_warps=4, num_stages=3,
    ),  # S2 真实 best
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 128, "GROUP_M": 1},
        num_warps=2, num_stages=3,
    ),  # S1 真实 best
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 256, "GROUP_M": 1},
        num_warps=2, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 256, "GROUP_M": 1},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 64, "GROUP_M": 1},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_M": 1},
        num_warps=2, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 1},
        num_warps=4, num_stages=3,
    ),

    # ============================================================
    # Medium M 配置 (M=64-256): GROUP_M 试 1, 4
    # ============================================================
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 1},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 4},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 1},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4},
        num_warps=4, num_stages=3,
    ),

    # ============================================================
    # Large M 配置 (M=512+, prefill): swizzle 收益最大
    # 每个 BLOCK 组合试 GROUP_M = 1, 4, 8, 16
    # ============================================================
    
    # BLOCK_M=128, BLOCK_N=128
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 1},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=4, num_stages=3,
    ),
    
    # BLOCK_M=128, BLOCK_N=256 (大 N, 适合大形状)
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 1},
        num_warps=8, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 4},
        num_warps=8, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=8, num_stages=3,
    ),
    
    # BLOCK_M=256, BLOCK_N=128
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 1},
        num_warps=8, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 4},
        num_warps=8, num_stages=3,
    ),
    
    # BLOCK_M=64, BLOCK_N=256 (适合"窄长"的 M 形状, 比如 M=512)
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 1},
        num_warps=8, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 4},
        num_warps=8, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
        num_warps=8, num_stages=3,
    ),
    
    # BLOCK_M=256, BLOCK_N=64
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 1},
        num_warps=8, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 4},
        num_warps=8, num_stages=3,
    ),
]


# ============================================================
# Kernel: v3 = v2.1 kernel + 头部 swizzle 计算
# ============================================================
# 唯一变化: 用 1D pid + swizzle 公式算出 pid_m, pid_n
# 其余部分 (offsets, K-loop, dequant, store) 和 v2.1 完全一样

@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["M", "N", "K"],
)
@triton.jit
def _w8a8_scaled_mm_kernel_swizzled(
    a_ptr, b_ptr, c_ptr,
    scale_a_ptr, scale_b_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,   # ★ v3 新增: swizzle 分组大小
):
    """
    Swizzled program ID 计算:
    
    默认 (row-major):
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        -> 同时跑的 program 在一行上跨 N 维度
    
    Swizzled (GROUP_M 分组):
        把 grid 划分成 GROUP_M 行 × num_pid_n 列 的"组"
        组内按 (列优先) 排列, 让同时跑的 ~GROUP_M*ceil(N/BLOCK_N)/SM 个
        program 集中在一个 GROUP_M × small_N 的区域里, B 在 L2 上的复用率提升.
    
    GROUP_M = 1 时, swizzle 公式退化为 row-major.
    """
    pid = tl.program_id(axis=0)   # 1D pid, 不再是 2D

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # ========== Swizzle 计算 (v3 核心新增, 6 行) ==========
    # 一组 = GROUP_M 行 × num_pid_n 列
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group         # 当前 program 在第几组
    first_pid_m = group_id * GROUP_M           # 这组的起始 M 行
    
    # 防止最后一组不满 GROUP_M (M 不是 GROUP_M * BLOCK_M 整数倍时)
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    # 组内, program 按 (列, 行) 排列
    # pid_m: 组内 M 偏移; pid_n: 组内 N 列
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    # ====================================================

    # ===== 以下完全和 v2.1 一样, 一字未改 =====
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
# Python wrapper: grid 改成 1D
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
    Triton W8A8 scaled matmul (v3: swizzled).
    
    相对 v2.1 的变化:
        1. autotune 候选加了 GROUP_M 维度 (1, 4, 8)
        2. Kernel 内部用 swizzle 公式重新计算 pid_m, pid_n
        3. Wrapper 的 grid 改为 1D (因为 kernel 内自己拆 pid)
    
    精度: 仍然 max_diff = 0 (算法没变, 只是 program 启动顺序变了)
    """
    assert a.dtype == torch.int8
    assert b.dtype == torch.int8
    assert scale_a.dtype == torch.float32
    assert scale_b.dtype == torch.float32
    assert a.is_cuda and b.is_cuda
    assert a.shape[1] == b.shape[0]
    assert bias is None, "v3 暂不支持 bias"
    assert out_dtype == torch.float16, "v3 暂只支持 fp16 输出"

    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert scale_a.shape == (M, 1)
    assert scale_b.shape == (1, N)

    c = torch.empty((M, N), dtype=out_dtype, device=a.device)

    # ★ v3 关键变化: 1D grid
    # v2.1: grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))   # 2D
    # v3:   grid = (cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N),) # 1D, kernel 内拆分
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )

    _w8a8_scaled_mm_kernel_swizzled[grid](
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
    print("Triton W8A8 v3 (swizzled, GROUP_M autotune) -- Smoke Test")
    print("=" * 70)

    torch.manual_seed(42)
    M, N, K = 16, 256, 256
    device = "cuda"

    x_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    print(f"\nNumber of autotune configs: {len(_AUTOTUNE_CONFIGS)}")
    print("First call (autotune will try all configs)...")
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
        print("FAIL — swizzle 改变了结果, 检查 swizzle 公式")

    # ----- 也测一下大形状 (swizzle 真正能发挥作用的地方) -----
    print()
    print("=" * 70)
    print("Quick check on S4-like shape (where swizzle should help)")
    print("=" * 70)
    M, N, K = 2048, 4096, 4096
    x_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    print(f"S4: M={M}, N={N}, K={K}")
    print("First call (autotune for S4)...")
    t0 = time.time()
    out_triton = w8a8_scaled_mm_triton(x_q, w_q, sa, sb)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"First call: {t1-t0:.2f}s")

    # warmup
    for _ in range(10):
        out_triton = w8a8_scaled_mm_triton(x_q, w_q, sa, sb)
    torch.cuda.synchronize()

    # 简短测量
    times = []
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out_triton = w8a8_scaled_mm_triton(x_q, w_q, sa, sb)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    median_us = times[len(times)//2] * 1000
    print(f"v3 S4 latency: {median_us:.2f} us (v2.1 was 190us)")

    out_ref = w8a8_scaled_mm_reference(x_q, w_q, sa, sb)
    max_diff = (out_triton.float() - out_ref.float()).abs().max().item()
    print(f"max_abs_diff: {max_diff:.4e}")