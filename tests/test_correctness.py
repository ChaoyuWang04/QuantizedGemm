"""
Correctness tests for W8A8 scaled matmul.

策略:
    1. Reference 自洽: 同一输入跑两次,结果完全一致 (no nondeterminism)
    2. 边界 case: 零、极值、补码、小形状、非对齐
    3. LLaMA 形状: 4 个标准形状的 sanity check

执行:
    pytest tests/test_correctness.py -v
    pytest tests/test_correctness.py::test_reference_zero -v   # 单个用例
"""

import pytest
import torch
import sys
from pathlib import Path

# Make reference module importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from reference.torch_reference import (
    w8a8_scaled_mm_reference,
    quantize_per_token,
    quantize_per_channel,
)


# Skip all GPU tests if no CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
DEVICE = "cuda"


# ============================================================
# Reference self-consistency
# ============================================================

def test_reference_deterministic():
    """Same input -> same output, twice."""
    torch.manual_seed(0)
    a = torch.randint(-128, 128, (16, 256), dtype=torch.int8, device=DEVICE)
    b = torch.randint(-128, 128, (256, 256), dtype=torch.int8, device=DEVICE)
    sa = torch.rand(16, 1, dtype=torch.float32, device=DEVICE) * 0.1
    sb = torch.rand(1, 256, dtype=torch.float32, device=DEVICE) * 0.01
    
    out1 = w8a8_scaled_mm_reference(a, b, sa, sb)
    out2 = w8a8_scaled_mm_reference(a, b, sa, sb)
    assert torch.equal(out1, out2), "Reference is non-deterministic!"


# ============================================================
# Edge cases
# ============================================================

def test_reference_zero_input():
    """All-zero activation -> all-zero output (modulo bias)."""
    a = torch.zeros((4, 64), dtype=torch.int8, device=DEVICE)
    b = torch.randint(-128, 128, (64, 64), dtype=torch.int8, device=DEVICE)
    sa = torch.ones(4, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.ones(1, 64, dtype=torch.float32, device=DEVICE)
    
    out = w8a8_scaled_mm_reference(a, b, sa, sb)
    assert torch.all(out == 0), f"Expected all zero, got max={out.abs().max()}"


def test_reference_max_input_no_overflow():
    """A=127, B=127, K=4096 -> int32 sum = 127*127*4096 ≈ 6.6e7, safe."""
    M, N, K = 8, 64, 4096
    a = torch.full((M, K), 127, dtype=torch.int8, device=DEVICE)
    b = torch.full((K, N), 127, dtype=torch.int8, device=DEVICE)
    sa = torch.ones(M, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.ones(1, N, dtype=torch.float32, device=DEVICE)
    
    out = w8a8_scaled_mm_reference(a, b, sa, sb, out_dtype=torch.float32)
    expected = 127.0 * 127.0 * K  # ≈ 6.6e7
    # fp32 can represent this exactly? actually 6.604e7 has only ~7 significant digits
    # so allow small relative error
    actual = out[0, 0].item()
    rel_err = abs(actual - expected) / expected
    assert rel_err < 1e-5, f"Expected ~{expected}, got {actual}, rel_err={rel_err}"


def test_reference_negative_input():
    """A=-128, B=-128 -> positive sum (two negatives)."""
    M, N, K = 4, 16, 64
    a = torch.full((M, K), -128, dtype=torch.int8, device=DEVICE)
    b = torch.full((K, N), -128, dtype=torch.int8, device=DEVICE)
    sa = torch.ones(M, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.ones(1, N, dtype=torch.float32, device=DEVICE)
    
    out = w8a8_scaled_mm_reference(a, b, sa, sb, out_dtype=torch.float32)
    expected = float((-128) * (-128) * K)  # exact
    assert torch.allclose(out, torch.full_like(out, expected))


def test_reference_small_shape():
    """Smallest viable shape."""
    M, N, K = 16, 16, 16
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEVICE)
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=DEVICE)
    sa = torch.rand(M, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.rand(1, N, dtype=torch.float32, device=DEVICE)
    
    out = w8a8_scaled_mm_reference(a, b, sa, sb)
    assert out.shape == (M, N) and out.dtype == torch.float16


def test_reference_non_aligned_shape():
    """Non-power-of-2 shape (real-world)."""
    M, N, K = 15, 33, 129
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEVICE)
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=DEVICE)
    sa = torch.rand(M, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.rand(1, N, dtype=torch.float32, device=DEVICE)
    
    out = w8a8_scaled_mm_reference(a, b, sa, sb)
    assert out.shape == (M, N)


def test_reference_with_bias():
    """Bias should be broadcast and added."""
    M, N, K = 8, 32, 32
    a = torch.zeros((M, K), dtype=torch.int8, device=DEVICE)  # 0 input
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=DEVICE)
    sa = torch.ones(M, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.ones(1, N, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(N, dtype=torch.float16, device=DEVICE)
    
    out = w8a8_scaled_mm_reference(a, b, sa, sb, bias=bias)
    # With zero activation, output should equal bias (broadcast)
    expected = bias.unsqueeze(0).expand(M, N)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


# ============================================================
# LLaMA-7B shapes: sanity check that nothing crashes
# ============================================================

@pytest.mark.parametrize("M,N,K", [
    pytest.param(1, 4096, 4096, id="S1_decode_b1"),
    pytest.param(16, 4096, 4096, id="S2_decode_b16"),
    pytest.param(512, 4096, 4096, id="S3_prefill_512"),
    pytest.param(2048, 4096, 4096, id="S4_prefill_2048"),
])
def test_reference_llama_shapes(M, N, K):
    """LLaMA-7B q/k/v/o projections."""
    torch.manual_seed(42)
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEVICE)
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=DEVICE)
    sa = torch.rand(M, 1, dtype=torch.float32, device=DEVICE) * 0.1
    sb = torch.rand(1, N, dtype=torch.float32, device=DEVICE) * 0.01
    
    out = w8a8_scaled_mm_reference(a, b, sa, sb)
    assert out.shape == (M, N)
    assert out.dtype == torch.float16
    assert not torch.isnan(out).any(), "NaN in output!"
    assert not torch.isinf(out).any(), "Inf in output!"


# ============================================================
# Quantization roundtrip sanity
# ============================================================

def test_quantize_roundtrip_per_token():
    """Quantize then dequantize should approximately recover input."""
    torch.manual_seed(0)
    x = torch.randn(32, 512, device=DEVICE, dtype=torch.float32)
    x_q, scale = quantize_per_token(x)
    x_recovered = x_q.float() * scale
    
    # Max error bounded by scale/2 (rounding error)
    max_err = (x - x_recovered).abs().max().item()
    max_scale = scale.max().item()
    assert max_err <= max_scale, f"max_err={max_err}, max_scale={max_scale}"


def test_quantize_roundtrip_per_channel():
    """Same for per-channel."""
    torch.manual_seed(0)
    w = torch.randn(512, 256, device=DEVICE, dtype=torch.float32) * 0.1
    w_q, scale = quantize_per_channel(w)
    w_recovered = w_q.float() * scale
    
    max_err = (w - w_recovered).abs().max().item()
    max_scale = scale.max().item()
    assert max_err <= max_scale


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
