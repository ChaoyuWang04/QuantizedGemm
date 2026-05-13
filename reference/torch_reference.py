"""
W8A8 scaled matmul 的 PyTorch 金标准实现。

这个文件用于给 Triton/CUTLASS kernel 提供正确性参考:先做 int32 矩阵乘累加,
再按 per-token/per-channel scale 反量化,最后可选加 bias 并转换输出 dtype。

数学定义:
    C[i,j] = (sum_k A_q[i,k] * B_q[k,j]) * scale_a[i] * scale_b[j] + bias[j]

实现要点:
    1. int8 先转 int32,避免 K 很大时累加溢出。
    2. int32 matmul 保持和硬件 int8 mma 累加语义一致。
    3. fp32 反量化和加 bias,最后再 cast 到目标 dtype。
"""

from typing import Optional
import torch


def w8a8_scaled_mm_reference(
    a: torch.Tensor,           # [M, K] int8
    b: torch.Tensor,           # [K, N] int8
    scale_a: torch.Tensor,     # [M, 1] fp32
    scale_b: torch.Tensor,     # [1, N] fp32
    bias: Optional[torch.Tensor] = None,  # [N] fp16
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    W8A8 scaled matmul 的慢速但严格参考实现。

    参数:
        a: int8 激活矩阵,[M, K]。
        b: int8 权重矩阵,[K, N]。
        scale_a: 激活 per-token scale,[M, 1],fp32。
        scale_b: 权重 per-channel scale,[1, N],fp32。
        bias: 可选 bias,[N]。
        out_dtype: 输出 dtype,默认 fp16。

    返回:
        形状为 [M, N] 的输出矩阵。
    """
    # 校验 dtype 和形状,保证后续广播和 matmul 的语义明确。
    assert a.dtype == torch.int8, f"a must be int8, got {a.dtype}"
    assert b.dtype == torch.int8, f"b must be int8, got {b.dtype}"
    assert scale_a.dtype == torch.float32, f"scale_a must be fp32, got {scale_a.dtype}"
    assert scale_b.dtype == torch.float32, f"scale_b must be fp32, got {scale_b.dtype}"
    assert a.shape[1] == b.shape[0], f"K mismatch: a {a.shape}, b {b.shape}"

    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert scale_a.shape == (M, 1), f"scale_a expected [{M}, 1], got {scale_a.shape}"
    assert scale_b.shape == (1, N), f"scale_b expected [1, {N}], got {scale_b.shape}"

    # int8 乘法累加必须放到 int32 域,否则 K 较大时会溢出。
    a_i32 = a.to(torch.int32)
    b_i32 = b.to(torch.int32)
    acc = torch.matmul(a_i32, b_i32)  # [M, N] int32

    # int32 累加结果转 fp32 后乘两个 scale,PyTorch 会自动广播 [M,1] 和 [1,N]。
    out_fp32 = acc.to(torch.float32) * scale_a * scale_b

    # bias 在 fp32 中相加,最后统一转成用户要求的输出 dtype。
    if bias is not None:
        out_fp32 = out_fp32 + bias.to(torch.float32)

    return out_fp32.to(out_dtype)


def quantize_per_token(x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对激活矩阵做对称 per-token 量化。

    每一行独立取 absmax,得到该 token 的 scale,再映射到 int8。

    参数:
        x: fp16/fp32 激活矩阵,[M, K]。
        eps: 防止全零行导致 scale 为 0。

    返回:
        (x_int8, scale_a),其中 scale_a 形状为 [M, 1]。
    """
    x_fp = x.to(torch.float32)
    # 每个 token/行单独求最大绝对值,对应 per-token scale。
    absmax = x_fp.abs().amax(dim=-1, keepdim=True).clamp(min=eps)  # [M, 1]
    scale_a = absmax / 127.0  # [M, 1]
    # round 后 clamp 到 int8 合法范围,完成对称量化。
    x_int8 = (x_fp / scale_a).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale_a


def quantize_per_channel(w: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对权重矩阵做对称 per-channel 量化。

    每一列对应一个输出 channel,独立计算 scale。

    参数:
        w: fp16/fp32 权重矩阵,[K, N]。
        eps: 防止全零列导致 scale 为 0。

    返回:
        (w_int8, scale_b),其中 scale_b 形状为 [1, N]。
    """
    w_fp = w.to(torch.float32)
    # 每个输出 channel/列单独求最大绝对值。
    absmax = w_fp.abs().amax(dim=0, keepdim=True).clamp(min=eps)  # [1, N]
    scale_b = absmax / 127.0  # [1, N]
    # 使用对应列的 scale 将权重量化到 int8。
    w_int8 = (w_fp / scale_b).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale_b


if __name__ == "__main__":
    # 简单冒烟测试:生成 fp16 输入,量化后跑 reference。
    torch.manual_seed(42)
    M, N, K = 16, 4096, 4096
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构造模拟激活和权重,权重乘 0.1 让数值范围更接近真实模型。
    x_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1

    # 分别按 per-token 和 per-channel 策略量化。
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    out = w8a8_scaled_mm_reference(x_q, w_q, sa, sb)
    print(f"Output shape: {out.shape}, dtype: {out.dtype}")
    print(f"Output stats: mean={out.mean():.4f}, std={out.std():.4f}, "
          f"absmax={out.abs().max():.4f}")

    # 和原始 fp16 matmul 对比,误差主要来自量化噪声。
    out_fp16_ref = torch.matmul(x_fp16, w_fp16)
    max_diff = (out.float() - out_fp16_ref.float()).abs().max().item()
    print(f"vs fp16 matmul: max_abs_diff={max_diff:.4f}")
    # 期望:误差非零但有限,大致反映 int8 量化损失。
