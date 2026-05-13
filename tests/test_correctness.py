"""
W8A8 scaled matmul 的正确性测试。

这个文件先验证 PyTorch reference 的确定性、边界输入和量化函数行为,
后续 Triton/CUTLASS 实现加入后也会以这里的 reference 为对齐目标。

测试策略:
    1. 同一输入跑两次,确认 reference 没有随机性。
    2. 覆盖零值、极值、负数、小 shape、非对齐 shape 和 bias。
    3. 跑 LLaMA-7B 的标准 shape,确认大尺寸不崩溃且没有 NaN/Inf。
"""

import pytest
import torch
import sys
from pathlib import Path

# 允许从 tests/ 目录直接导入仓库根目录下的 reference 模块。
sys.path.insert(0, str(Path(__file__).parent.parent))
from reference.torch_reference import (
    w8a8_scaled_mm_reference,
    quantize_per_token,
    quantize_per_channel,
)


# 当前测试全部跑在 CUDA 上;没有 GPU 时整体跳过。
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
DEVICE = "cuda"


# ============================================================
# Reference self-consistency
# ============================================================

def test_reference_deterministic():
    """同一份输入连续运行两次,输出必须完全一致。"""
    torch.manual_seed(0)
    # 固定 seed 生成输入,用于排查 reference 是否有非确定行为。
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
    """激活全 0 时,无 bias 输出也应全 0。"""
    # a 全零会让 int32 累加结果全零,用于验证 scale 不引入额外值。
    a = torch.zeros((4, 64), dtype=torch.int8, device=DEVICE)
    b = torch.randint(-128, 128, (64, 64), dtype=torch.int8, device=DEVICE)
    sa = torch.ones(4, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.ones(1, 64, dtype=torch.float32, device=DEVICE)

    out = w8a8_scaled_mm_reference(a, b, sa, sb)
    assert torch.all(out == 0), f"Expected all zero, got max={out.abs().max()}"


def test_reference_max_input_no_overflow():
    """正向极值输入用于验证 int32 累加不会溢出。"""
    M, N, K = 8, 64, 4096
    # 127*127*K 仍远小于 int32 上限,但会超过 int16。
    a = torch.full((M, K), 127, dtype=torch.int8, device=DEVICE)
    b = torch.full((K, N), 127, dtype=torch.int8, device=DEVICE)
    sa = torch.ones(M, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.ones(1, N, dtype=torch.float32, device=DEVICE)

    out = w8a8_scaled_mm_reference(a, b, sa, sb, out_dtype=torch.float32)
    expected = 127.0 * 127.0 * K  # ≈ 6.6e7
    # fp32 只有约 7 位有效数字,这里用相对误差而不是完全相等。
    actual = out[0, 0].item()
    rel_err = abs(actual - expected) / expected
    assert rel_err < 1e-5, f"Expected ~{expected}, got {actual}, rel_err={rel_err}"


def test_reference_negative_input():
    """负向极值相乘应得到正数,用于验证补码符号处理。"""
    M, N, K = 4, 16, 64
    a = torch.full((M, K), -128, dtype=torch.int8, device=DEVICE)
    b = torch.full((K, N), -128, dtype=torch.int8, device=DEVICE)
    sa = torch.ones(M, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.ones(1, N, dtype=torch.float32, device=DEVICE)

    out = w8a8_scaled_mm_reference(a, b, sa, sb, out_dtype=torch.float32)
    expected = float((-128) * (-128) * K)  # exact
    assert torch.allclose(out, torch.full_like(out, expected))


def test_reference_small_shape():
    """最小可运行 shape,用于快速覆盖基础路径。"""
    M, N, K = 16, 16, 16
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEVICE)
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=DEVICE)
    sa = torch.rand(M, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.rand(1, N, dtype=torch.float32, device=DEVICE)

    out = w8a8_scaled_mm_reference(a, b, sa, sb)
    assert out.shape == (M, N) and out.dtype == torch.float16


def test_reference_non_aligned_shape():
    """非 2 的幂且非对齐 shape,用于提前暴露 shape 假设问题。"""
    M, N, K = 15, 33, 129
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEVICE)
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=DEVICE)
    sa = torch.rand(M, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.rand(1, N, dtype=torch.float32, device=DEVICE)

    out = w8a8_scaled_mm_reference(a, b, sa, sb)
    assert out.shape == (M, N)


def test_reference_with_bias():
    """bias 应沿 batch/token 维广播并加到每一行。"""
    M, N, K = 8, 32, 32
    a = torch.zeros((M, K), dtype=torch.int8, device=DEVICE)  # 0 input
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=DEVICE)
    sa = torch.ones(M, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.ones(1, N, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(N, dtype=torch.float16, device=DEVICE)

    out = w8a8_scaled_mm_reference(a, b, sa, sb, bias=bias)
    # 激活为 0 时,矩阵乘部分为 0,输出应只剩 bias 的广播结果。
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
    """LLaMA-7B 投影层典型 shape 的大尺寸冒烟测试。"""
    torch.manual_seed(42)
    # 使用需求文档中的随机分布,保证 shape 和数值范围都接近目标场景。
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEVICE)
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=DEVICE)
    sa = torch.rand(M, 1, dtype=torch.float32, device=DEVICE) * 0.1
    sb = torch.rand(1, N, dtype=torch.float32, device=DEVICE) * 0.01

    out = w8a8_scaled_mm_reference(a, b, sa, sb)
    # 这里不和其他 kernel 对比,只确认 reference 在目标 shape 上稳定产出。
    assert out.shape == (M, N)
    assert out.dtype == torch.float16
    assert not torch.isnan(out).any(), "NaN in output!"
    assert not torch.isinf(out).any(), "Inf in output!"


# ============================================================
# Quantization roundtrip sanity
# ============================================================

def test_quantize_roundtrip_per_token():
    """per-token 量化再反量化后,误差应被 scale 量级约束。"""
    torch.manual_seed(0)
    x = torch.randn(32, 512, device=DEVICE, dtype=torch.float32)
    x_q, scale = quantize_per_token(x)
    x_recovered = x_q.float() * scale

    # round 造成的最大误差应不超过单个量化 step 的量级。
    max_err = (x - x_recovered).abs().max().item()
    max_scale = scale.max().item()
    assert max_err <= max_scale, f"max_err={max_err}, max_scale={max_scale}"


def test_quantize_roundtrip_per_channel():
    """per-channel 权重量化的 roundtrip 误差检查。"""
    torch.manual_seed(0)
    w = torch.randn(512, 256, device=DEVICE, dtype=torch.float32) * 0.1
    w_q, scale = quantize_per_channel(w)
    w_recovered = w_q.float() * scale

    # 逐列 scale 不同,因此用全局最大 scale 给最大误差设上界。
    max_err = (w - w_recovered).abs().max().item()
    max_scale = scale.max().item()
    assert max_err <= max_scale


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
