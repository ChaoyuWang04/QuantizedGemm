"""
PyTorch reference implementation for W8A8 scaled matmul.

This is the GOLD STANDARD. All Triton/CUTLASS implementations must align with this.

数学定义:
    C[i,j] = (sum_k A_q[i,k] * B_q[k,j]) * scale_a[i] * scale_b[j] + bias[j]
    
    where:
        A_q in int8, shape [M, K]
        B_q in int8, shape [K, N]
        scale_a in fp32, shape [M, 1]
        scale_b in fp32, shape [1, N]
        bias   in fp16, shape [N]    (optional)
        C      in fp16, shape [M, N]

实现策略:
    1. int8 -> int32 cast (避免溢出)
    2. int32 matmul (PyTorch native, uses CPU/GPU based on device)
    3. cast to fp32, multiply by scales (broadcasting)
    4. add bias, cast to target dtype

为什么用 int32 中间结果:
    int8 max = 127, K=4096 时 sum 上界 = 127 * 127 * 4096 ≈ 6.6e7
    int32 max ≈ 2.1e9, 安全
    int16 max = 32767, 不够

为什么不直接 cast 成 fp32 算:
    fp32 matmul 有 rounding error, 不是"金标准"
    int32 matmul 在数学上严格等价于 int8 mma 的累加结果
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
    Reference implementation. SLOW BUT CORRECT.
    
    Args:
        a: int8 activation tensor [M, K]
        b: int8 weight tensor [K, N]
        scale_a: fp32 per-token scale [M, 1]
        scale_b: fp32 per-channel scale [1, N]
        bias: optional fp16 bias [N]
        out_dtype: output dtype, default fp16
    
    Returns:
        Output tensor [M, N] in out_dtype.
    """
    # ===== Input validation =====
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
    
    # ===== Core computation =====
    # int32 accumulation
    a_i32 = a.to(torch.int32)
    b_i32 = b.to(torch.int32)
    acc = torch.matmul(a_i32, b_i32)  # [M, N] int32
    
    # Dequantize: int32 -> fp32 -> apply scales
    out_fp32 = acc.to(torch.float32) * scale_a * scale_b
    
    # Bias (in fp32 for precision, cast later)
    if bias is not None:
        out_fp32 = out_fp32 + bias.to(torch.float32)
    
    return out_fp32.to(out_dtype)


def quantize_per_token(x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-token quantization for activations.
    
    Args:
        x: fp16/fp32 tensor [M, K]
        eps: numerical stability
    
    Returns:
        (x_int8, scale_a) where scale_a has shape [M, 1]
    """
    x_fp = x.to(torch.float32)
    # absmax per row
    absmax = x_fp.abs().amax(dim=-1, keepdim=True).clamp(min=eps)  # [M, 1]
    scale_a = absmax / 127.0  # [M, 1]
    x_int8 = (x_fp / scale_a).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale_a


def quantize_per_channel(w: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-channel quantization for weights.
    
    Args:
        w: fp16/fp32 tensor [K, N]
        eps: numerical stability
    
    Returns:
        (w_int8, scale_b) where scale_b has shape [1, N]
    """
    w_fp = w.to(torch.float32)
    # absmax per column
    absmax = w_fp.abs().amax(dim=0, keepdim=True).clamp(min=eps)  # [1, N]
    scale_b = absmax / 127.0  # [1, N]
    w_int8 = (w_fp / scale_b).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale_b


if __name__ == "__main__":
    # Smoke test
    torch.manual_seed(42)
    M, N, K = 16, 4096, 4096
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate fake fp16 inputs and quantize
    x_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1
    
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)
    
    out = w8a8_scaled_mm_reference(x_q, w_q, sa, sb)
    print(f"Output shape: {out.shape}, dtype: {out.dtype}")
    print(f"Output stats: mean={out.mean():.4f}, std={out.std():.4f}, "
          f"absmax={out.abs().max():.4f}")
    
    # Sanity: compare with naive fp16 matmul
    out_fp16_ref = torch.matmul(x_fp16, w_fp16)
    max_diff = (out.float() - out_fp16_ref.float()).abs().max().item()
    print(f"vs fp16 matmul: max_abs_diff={max_diff:.4f}")
    # Expected: small but nonzero (quantization noise ~1e-1 scale)
