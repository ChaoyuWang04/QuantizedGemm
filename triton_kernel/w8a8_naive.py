"""
W8A8 scaled matmul 的 Triton kernel 入口占位文件。

当前项目还处在 P1/P2 交界,这里先记录 Triton 实现计划,并显式抛出
NotImplementedError,避免调用方误以为 kernel 已可用。

计划:
    v1: 朴素 Triton matmul,手动 int32 累加并应用 scale。
    v2: 加 @triton.autotune 搜索 BLOCK_M/BLOCK_N/BLOCK_K 等参数。
    v3: 根据 Nsight 结果优化访存、swizzle 和可能的 split-K。

参考:
    https://huggingface.co/kernels-community/triton-scaled-mm
    https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/w8a8_utils.py
"""

"""
W8A8 scaled matmul - Triton kernel 实现 (v1: naive 版本)

设计目标 (P2 阶段第一步):
    1. 能编译, 能跑
    2. 精度对齐 PyTorch reference (max_abs_diff < 1e-2)
    3. 暂不考虑性能 —— 性能优化是 v2/v3 的工作

不在 v1 实现的 (留给 v2+):
    - @triton.autotune (BLOCK_M/N/K 调参)
    - swizzle / 矩阵 tile 的 super-grouping (改善 L2 缓存命中)
    - split-K (K 维度并行)
    - 异步 pipelining
    - bias 支持 (先把核心路径跑通)

数学定义 (复习):
    acc[m,n] = sum_k a_q[m,k] * b_q[k,n]                    # int8×int8 → int32
    c[m,n]   = acc[m,n] * scale_a[m] * scale_b[n]            # int32 → fp32 → fp16

Tile 设计:
    - 每个 program 负责输出矩阵 C 的一个 [BLOCK_M, BLOCK_N] tile
    - 沿 K 维度循环, 每次加载 [BLOCK_M, BLOCK_K] 的 A tile 和 [BLOCK_K, BLOCK_N] 的 B tile
    - K-loop 内累加器保持 int32 (绝对不做 dequant)
    - K-loop 结束后, 一次性应用 scale_a × scale_b 反量化
"""

from typing import Optional
import torch
import triton
import triton.language as tl


# ============================================================
# Triton kernel: 在 GPU 上跑的真正核心
# ============================================================

@triton.jit
def _w8a8_scaled_mm_kernel(
    # ----- 数据指针 -----
    a_ptr, b_ptr, c_ptr,
    scale_a_ptr, scale_b_ptr,
    # ----- 矩阵维度 -----
    M, N, K,
    # ----- Stride: 矩阵在每个维度上的步长 (元素数, 不是字节) -----
    # 对于 row-major 的 a [M, K]: stride_am = K, stride_ak = 1
    # 对于 row-major 的 b [K, N]: stride_bk = N, stride_bn = 1
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # ----- Block size (编译期常量, 由调用方指定) -----
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    每个 program 负责计算输出矩阵 C 的一个 [BLOCK_M, BLOCK_N] tile.

    Grid 设计:
        program_id_0 -> tile 的行索引 (M 方向)
        program_id_1 -> tile 的列索引 (N 方向)
        总 program 数 = ceil(M/BLOCK_M) × ceil(N/BLOCK_N)
    """
    # ============================================================
    # 第 1 步: 算出当前 program 负责哪个 tile
    # ============================================================
    pid_m = tl.program_id(axis=0)   # 当前 program 的行 tile 索引
    pid_n = tl.program_id(axis=1)   # 当前 program 的列 tile 索引

    # ============================================================
    # 第 2 步: 构造 offsets (本 program 要读/写的具体元素下标)
    # ============================================================
    # offs_m: [BLOCK_M] 的向量, 表示这个 tile 在 M 维度的全局行号
    # 例如 pid_m=0, BLOCK_M=64 -> offs_m = [0, 1, ..., 63]
    # 例如 pid_m=2, BLOCK_M=64 -> offs_m = [128, 129, ..., 191]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # K 方向的 offset 在 K-loop 内会更新, 这里先初始化为 0..BLOCK_K
    offs_k = tl.arange(0, BLOCK_K)

    # ============================================================
    # 第 3 步: 构造 pointer 矩阵 (A tile 和 B tile 的具体地址)
    # ============================================================
    # A 的元素 [m, k] 地址 = a_ptr + m * stride_am + k * stride_ak
    # 用 broadcasting 构造 [BLOCK_M, BLOCK_K] 的地址矩阵:
    #   offs_m[:, None]: [BLOCK_M, 1]
    #   offs_k[None, :]: [1, BLOCK_K]
    #   相加广播成     : [BLOCK_M, BLOCK_K]
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # ============================================================
    # 第 4 步: K-loop —— GEMM 的灵魂
    # ============================================================
    # 累加器: int32 类型, 必须放在 register (不写 dtype 时默认 fp32, 会错)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # 每次循环处理 K 维的一个 BLOCK_K 窄条
    for k_start in range(0, K, BLOCK_K):
        # ---- 4a. 处理 K 边界 (K 不是 BLOCK_K 整数倍时) ----
        # mask: True 表示这个位置是合法数据, False 表示超出 K 范围 (要填 0)
        k_mask = offs_k[None, :] < (K - k_start)            # [1, BLOCK_K]
        a_mask = k_mask & (offs_m[:, None] < M)             # [BLOCK_M, BLOCK_K]
        b_mask = (offs_k[:, None] < (K - k_start)) & (offs_n[None, :] < N)

        # ---- 4b. 加载 A tile 和 B tile (越界位置填 0) ----
        # other=0: 越界元素当作 0 处理, 0 × 任何数 = 0, 不影响累加结果
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0)   # [BLOCK_M, BLOCK_K] int8
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0)   # [BLOCK_K, BLOCK_N] int8

        # ---- 4c. 矩阵乘累加 ----
        # tl.dot 是 Triton 的张量乘法, 会自动编译成 Tensor Core mma 指令
        # out_dtype=tl.int32 告诉编译器: 输入是 int8, 累加器是 int32
        # 这一行编译后是 mma.sync.aligned.m16n8k32.s32.s8.s8.s32 指令
        accumulator += tl.dot(a_tile, b_tile, out_dtype=tl.int32)

        # ---- 4d. 移动指针到下一个 K 窄条 ----
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ============================================================
    # 第 5 步: K-loop 结束, 应用 scale 反量化
    # ============================================================
    # 加载 scale (注意 mask, 防越界)
    # scale_a: [M, 1] -> 这个 tile 用 [BLOCK_M] 个 scale
    # scale_b: [1, N] -> 这个 tile 用 [BLOCK_N] 个 scale
    sa_mask = offs_m < M
    sb_mask = offs_n < N
    sa = tl.load(scale_a_ptr + offs_m, mask=sa_mask, other=0.0)  # [BLOCK_M]
    sb = tl.load(scale_b_ptr + offs_n, mask=sb_mask, other=0.0)  # [BLOCK_N]

    # int32 -> fp32, 然后乘 scale (用 broadcasting)
    c_fp32 = accumulator.to(tl.float32) * sa[:, None] * sb[None, :]

    # ============================================================
    # 第 6 步: 写回 C
    # ============================================================
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # cast 到 fp16 再写回 (kernel 内部全程 fp32, 输出 dtype 在外层指定)
    tl.store(c_ptrs, c_fp32.to(tl.float16), mask=c_mask)


# ============================================================
# Python wrapper: 调用 kernel, 处理 grid 计算和参数准备
# ============================================================

def w8a8_scaled_mm_triton(
    a: torch.Tensor,           # [M, K] int8
    b: torch.Tensor,           # [K, N] int8
    scale_a: torch.Tensor,     # [M, 1] fp32
    scale_b: torch.Tensor,     # [1, N] fp32
    bias: Optional[torch.Tensor] = None,  # [N], v1 暂不支持
    out_dtype: torch.dtype = torch.float16,
    # Block size: v1 用固定值, v2 改为 autotune
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 64,
) -> torch.Tensor:
    """
    Triton W8A8 scaled matmul (v1 naive)

    精度目标: max_abs_diff < 1e-2 vs w8a8_scaled_mm_reference
    性能目标: 暂无, v1 先跑通

    参数:
        a:       int8 激活矩阵 [M, K]
        b:       int8 权重矩阵 [K, N]
        scale_a: fp32 per-token scale [M, 1]
        scale_b: fp32 per-channel scale [1, N]
        bias:    暂不支持 (v1)
        out_dtype: 输出 dtype, v1 只支持 fp16

    返回:
        [M, N] out_dtype 的输出张量
    """
    # ----- 输入校验 -----
    assert a.dtype == torch.int8, f"a must be int8, got {a.dtype}"
    assert b.dtype == torch.int8, f"b must be int8, got {b.dtype}"
    assert scale_a.dtype == torch.float32, f"scale_a must be fp32, got {scale_a.dtype}"
    assert scale_b.dtype == torch.float32, f"scale_b must be fp32, got {scale_b.dtype}"
    assert a.is_cuda and b.is_cuda, "Triton kernel 只在 CUDA 上跑"
    assert a.shape[1] == b.shape[0], f"K mismatch: a {a.shape}, b {b.shape}"

    # v1 限制: 暂不支持 bias, 暂只支持 fp16 输出
    assert bias is None, "v1 暂不支持 bias, 会在 v2 加上"
    assert out_dtype == torch.float16, "v1 暂只支持 fp16 输出"

    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert scale_a.shape == (M, 1)
    assert scale_b.shape == (1, N)

    # ----- 准备输出 buffer -----
    c = torch.empty((M, N), dtype=out_dtype, device=a.device)

    # ----- 准备 scale (kernel 内是 1D 访问, 这里 squeeze 一下) -----
    # scale_a: [M, 1] -> [M], scale_b: [1, N] -> [N]
    sa_1d = scale_a.squeeze(-1).contiguous()
    sb_1d = scale_b.squeeze(0).contiguous()

    # ----- 算 grid: 启动多少个 program -----
    # cdiv(x, y) = ceil(x / y), 上取整, 防止 M/N 不是 BLOCK 整数倍时漏掉
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # ----- 启动 kernel -----
    _w8a8_scaled_mm_kernel[grid](
        a, b, c,
        sa_1d, sb_1d,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return c


# ============================================================
# Smoke test: 单文件可运行, 自我验证
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
    print("Triton W8A8 v1 -- Smoke Test")
    print("=" * 70)

    torch.manual_seed(42)
    M, N, K = 16, 256, 256   # 小形状先跑通, 不爆显存
    device = "cuda"

    # 构造输入
    x_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    # 跑 Triton
    out_triton = w8a8_scaled_mm_triton(x_q, w_q, sa, sb)
    print(f"\n[Triton] shape={out_triton.shape}, dtype={out_triton.dtype}")
    print(f"         mean={out_triton.mean():.4f}, std={out_triton.std():.4f}, "
          f"absmax={out_triton.abs().max():.4f}")

    # 跑 reference 对照
    out_ref = w8a8_scaled_mm_reference(x_q, w_q, sa, sb)
    print(f"\n[Reference] shape={out_ref.shape}, dtype={out_ref.dtype}")
    print(f"            mean={out_ref.mean():.4f}, std={out_ref.std():.4f}, "
          f"absmax={out_ref.abs().max():.4f}")

    # 对比
    max_diff = (out_triton.float() - out_ref.float()).abs().max().item()
    rel_diff = max_diff / out_ref.abs().max().item()
    print(f"\n[Comparison]")
    print(f"  max_abs_diff: {max_diff:.4e}")
    print(f"  rel_diff:     {rel_diff*100:.3f}%")

    target_abs = 1e-2
    if max_diff < target_abs:
        print(f"  PASS (< {target_abs} target)")
    else:
        print(f"  FAIL (>= {target_abs} target)")
        print(f"  Debug hint: check input range, accumulator dtype, scale broadcasting")