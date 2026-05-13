"""
W8A8 scaled matmul 参考实现的正确性测试。

策略 (按测试金字塔自下而上):
    1. 量化辅助函数: roundtrip sanity
    2. 主 reference (int32 累加): 自洽 + 边界 case
    3. Ultra reference (fp64 累加): 自洽
    4. 跨实现一致性: int32 路径 vs fp64 路径必须严格对齐 (这是最重要的测试)
    5. fake_quantize: 量化方案探针的正确性
    6. cublas baseline: 性能对手的输出 dtype 正确性
    7. LLaMA-7B 4 个形状: sanity check

执行:
    pytest tests/test_correctness.py -v                            # 全部
    pytest tests/test_correctness.py::test_cross_int32_vs_fp64 -v  # 单个
    pytest tests/test_correctness.py -v -k "not llama"             # 跳过慢测试
"""

import pytest
import torch
import sys
from pathlib import Path

# 让 reference 模块可被 import
sys.path.insert(0, str(Path(__file__).parent.parent))
from reference.torch_reference import (
    w8a8_scaled_mm_reference,
    w8a8_scaled_mm_reference_fp64,
    fake_quantize_linear,
    cublas_fp16_baseline,
    quantize_per_token,
    quantize_per_channel,
)


# ============================================================
# CUDA 检测: 区分必须 GPU 的测试 vs 可以 CPU 跑的测试
# ============================================================
# 量化函数和 reference 本身在 CPU 也能跑 (只是慢), 不应该被 skip。
# 但 cublas_fp16_baseline 在 CPU 上没有 cuBLAS, 行为不同, 必须 skip。

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")


# ============================================================
# 1. 量化辅助函数 roundtrip sanity
# ============================================================

def test_quantize_per_token_roundtrip():
    """量化 -> 反量化 应该近似还原, 误差上界 = max_scale (1 ULP)."""
    torch.manual_seed(0)
    x = torch.randn(32, 512, device=DEVICE, dtype=torch.float32)
    x_q, scale = quantize_per_token(x)
    x_recovered = x_q.float() * scale

    max_err = (x - x_recovered).abs().max().item()
    max_scale = scale.max().item()
    assert max_err <= max_scale, f"max_err={max_err}, max_scale={max_scale}"


def test_quantize_per_channel_roundtrip():
    """同上, 但按列."""
    torch.manual_seed(0)
    w = torch.randn(512, 256, device=DEVICE, dtype=torch.float32) * 0.1
    w_q, scale = quantize_per_channel(w)
    w_recovered = w_q.float() * scale

    max_err = (w - w_recovered).abs().max().item()
    max_scale = scale.max().item()
    assert max_err <= max_scale


def test_quantize_per_token_shapes_and_dtypes():
    """量化函数返回的 shape 和 dtype 必须严格正确."""
    x = torch.randn(8, 64, device=DEVICE, dtype=torch.float16)
    x_q, scale = quantize_per_token(x)
    assert x_q.shape == (8, 64)
    assert x_q.dtype == torch.int8
    assert scale.shape == (8, 1)
    assert scale.dtype == torch.float32


def test_quantize_per_channel_shapes_and_dtypes():
    """同上."""
    w = torch.randn(64, 32, device=DEVICE, dtype=torch.float16)
    w_q, scale = quantize_per_channel(w)
    assert w_q.shape == (64, 32)
    assert w_q.dtype == torch.int8
    assert scale.shape == (1, 32)
    assert scale.dtype == torch.float32


# ============================================================
# 2. 主 reference (int32 累加): 自洽 + 边界 case
# ============================================================

def test_reference_deterministic():
    """相同输入 -> 相同输出, 跑两次."""
    torch.manual_seed(0)
    a = torch.randint(-128, 128, (16, 256), dtype=torch.int8, device=DEVICE)
    b = torch.randint(-128, 128, (256, 256), dtype=torch.int8, device=DEVICE)
    sa = torch.rand(16, 1, dtype=torch.float32, device=DEVICE) * 0.1
    sb = torch.rand(1, 256, dtype=torch.float32, device=DEVICE) * 0.01

    out1 = w8a8_scaled_mm_reference(a, b, sa, sb)
    out2 = w8a8_scaled_mm_reference(a, b, sa, sb)
    assert torch.equal(out1, out2), "Reference is non-deterministic!"


def test_reference_zero_input():
    """A 全零 -> 输出全零 (无 bias 时)."""
    a = torch.zeros((4, 64), dtype=torch.int8, device=DEVICE)
    b = torch.randint(-128, 128, (64, 64), dtype=torch.int8, device=DEVICE)
    sa = torch.ones(4, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.ones(1, 64, dtype=torch.float32, device=DEVICE)

    out = w8a8_scaled_mm_reference(a, b, sa, sb)
    assert torch.all(out == 0), f"Expected all zero, got max={out.abs().max()}"


def test_reference_max_input_no_overflow():
    """
    极值输入: A=127, B=127, K=4096.
    
    int8 累加最大值: 127*127*4096 = 66,064,384
    
    精度分析:
        - CPU 路径 (int32 累加): 精确, 输出严格 = 66064384
        - GPU 路径 (fp32 累加): fp32 在 6.6e7 量级时 ULP ≈ 8,
          所以可能有 ~8 的 absolute error, 相对误差 ~1.2e-7
        - 两条路径都远不到 int32 上限 (2.15e9), 主要风险是 fp32 ULP
    
    所以这个测试容忍度按 fp32 ULP 设置, atol=100 (~12 ULP 余量) 足够。
    """
    M, N, K = 8, 64, 4096
    a = torch.full((M, K), 127, dtype=torch.int8, device=DEVICE)
    b = torch.full((K, N), 127, dtype=torch.int8, device=DEVICE)
    sa = torch.ones(M, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.ones(1, N, dtype=torch.float32, device=DEVICE)

    # 用 fp64 输出避开输出阶段的 ULP
    out = w8a8_scaled_mm_reference(a, b, sa, sb, out_dtype=torch.float64)
    expected = 127.0 * 127.0 * K  # 严格 = 66,064,384
    assert torch.allclose(out, torch.full_like(out, expected), atol=100.0), \
        f"Expected {expected}, got {out[0, 0].item()}"


def test_reference_negative_input():
    """
    A=-128, B=-128 -> 结果是精确正整数 (两负相乘为正)。
    
    128*128*64 = 1,048,576 = 2^20, fp32 / fp64 都能精确表示。
    GPU 上 fp32 累加可能因 reduction 顺序有 1 ULP 差异, 但应该几乎无误差。
    """
    M, N, K = 4, 16, 64
    a = torch.full((M, K), -128, dtype=torch.int8, device=DEVICE)
    b = torch.full((K, N), -128, dtype=torch.int8, device=DEVICE)
    sa = torch.ones(M, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.ones(1, N, dtype=torch.float32, device=DEVICE)

    out = w8a8_scaled_mm_reference(a, b, sa, sb, out_dtype=torch.float32)
    expected = float(128 * 128 * K)
    # K=64 小, fp32 累加足够精确, atol=1 = 1 个累加项的最大可能 ULP
    torch.testing.assert_close(out, torch.full_like(out, expected),
                                rtol=0, atol=1.0)


def test_reference_small_shape():
    """最小可运行形状, 防止有 hardcode 的 size 假设."""
    M, N, K = 16, 16, 16
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEVICE)
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=DEVICE)
    sa = torch.rand(M, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.rand(1, N, dtype=torch.float32, device=DEVICE)

    out = w8a8_scaled_mm_reference(a, b, sa, sb)
    assert out.shape == (M, N) and out.dtype == torch.float16


def test_reference_non_aligned_shape():
    """非 2 的幂形状, 真实模型中很常见 (例如 GQA 后的 head dim)."""
    M, N, K = 15, 33, 129
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEVICE)
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=DEVICE)
    sa = torch.rand(M, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.rand(1, N, dtype=torch.float32, device=DEVICE)

    out = w8a8_scaled_mm_reference(a, b, sa, sb)
    assert out.shape == (M, N)


def test_reference_with_bias():
    """A 全零时, 输出 = bias 广播."""
    M, N, K = 8, 32, 32
    a = torch.zeros((M, K), dtype=torch.int8, device=DEVICE)
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=DEVICE)
    sa = torch.ones(M, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.ones(1, N, dtype=torch.float32, device=DEVICE)
    bias = torch.randn(N, dtype=torch.float16, device=DEVICE)

    out = w8a8_scaled_mm_reference(a, b, sa, sb, bias=bias)
    expected = bias.unsqueeze(0).expand(M, N)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


def test_reference_input_validation():
    """非法输入应该抛 assertion error, 而不是产生错误结果."""
    a = torch.zeros((4, 64), dtype=torch.float16, device=DEVICE)  # 错误 dtype
    b = torch.zeros((64, 64), dtype=torch.int8, device=DEVICE)
    sa = torch.ones(4, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.ones(1, 64, dtype=torch.float32, device=DEVICE)
    with pytest.raises(AssertionError, match="a must be int8"):
        w8a8_scaled_mm_reference(a, b, sa, sb)


# ============================================================
# 3. Ultra reference (fp64 累加) 自洽
# ============================================================

def test_reference_fp64_deterministic():
    """fp64 reference 也要确定性."""
    torch.manual_seed(0)
    a = torch.randint(-128, 128, (8, 128), dtype=torch.int8, device=DEVICE)
    b = torch.randint(-128, 128, (128, 64), dtype=torch.int8, device=DEVICE)
    sa = torch.rand(8, 1, dtype=torch.float32, device=DEVICE) * 0.1
    sb = torch.rand(1, 64, dtype=torch.float32, device=DEVICE) * 0.01

    out1 = w8a8_scaled_mm_reference_fp64(a, b, sa, sb)
    out2 = w8a8_scaled_mm_reference_fp64(a, b, sa, sb)
    assert torch.equal(out1, out2)


# ============================================================
# 4. ★ 跨实现一致性: int32 路径 vs fp64 路径 (最重要的测试)
# ============================================================

@pytest.mark.parametrize("M,N,K", [
    pytest.param(16, 64, 256, id="small"),
    pytest.param(64, 128, 1024, id="medium"),
    pytest.param(16, 4096, 4096, id="llama_decode"),
])
def test_cross_int32_vs_fp64_random(M, N, K):
    """
    int32 累加路径 和 fp64 累加路径 必须严格对齐。
    
    用 fp64 输出避开 fp16/fp32 ULP rounding, 这样 diff 就只反映两个累加
    路径的差异。int32 是精确累加 (不溢出时), fp64 也是精确累加, 所以
    diff 应该极小 (1e-6 量级, 由 fp32 scale 乘法引入).
    """
    torch.manual_seed(42)
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device=DEVICE)
    b = torch.randint(-128, 128, (K, N), dtype=torch.int8, device=DEVICE)
    sa = torch.rand(M, 1, dtype=torch.float32, device=DEVICE) * 0.1
    sb = torch.rand(1, N, dtype=torch.float32, device=DEVICE) * 0.01

    out_int32 = w8a8_scaled_mm_reference(a, b, sa, sb, out_dtype=torch.float64)
    out_fp64 = w8a8_scaled_mm_reference_fp64(a, b, sa, sb, out_dtype=torch.float64)

    # 容忍度很严: 只允许 fp32 scale 乘法的 1 ULP 误差
    max_diff = (out_int32 - out_fp64).abs().max().item()
    # absmax 用来算相对误差
    absmax = out_fp64.abs().max().item() + 1e-9
    rel_err = max_diff / absmax
    assert rel_err < 1e-5, \
        f"int32 vs fp64 mismatch: abs={max_diff:.2e}, rel={rel_err:.2e} " \
        f"(shape M={M},N={N},K={K})"


def test_cross_int32_vs_fp64_extreme_values():
    """极值边界: 全 ±127, 验证 int32 即使逼近上限也精确."""
    M, N, K = 4, 32, 4096
    # 全 127
    a = torch.full((M, K), 127, dtype=torch.int8, device=DEVICE)
    b = torch.full((K, N), 127, dtype=torch.int8, device=DEVICE)
    sa = torch.ones(M, 1, dtype=torch.float32, device=DEVICE)
    sb = torch.ones(1, N, dtype=torch.float32, device=DEVICE)

    out_int32 = w8a8_scaled_mm_reference(a, b, sa, sb, out_dtype=torch.float64)
    out_fp64 = w8a8_scaled_mm_reference_fp64(a, b, sa, sb, out_dtype=torch.float64)
    # 都精确等于 127*127*4096 = 66064384
    expected = 127.0 * 127.0 * K
    assert torch.all(out_int32 == expected)
    assert torch.all(out_fp64 == expected)


# ============================================================
# 5. fake_quantize_linear: 量化方案探针
# ============================================================

@requires_cuda
def test_fake_quantize_matches_int32_reference():
    """
    fake_quantize_linear (fp32 matmul 后的量化-反量化结果) 应该
    和 w8a8_scaled_mm_reference (int32 真算) 数值接近。
    
    两者数学等价 (在精确算术下), 实际差异来自:
        - quantize 函数的 rounding (相同, 因为复用了量化函数)
        - matmul 路径不同 (int32 vs fp32), fp32 累加有 rounding
    
    diff 上界估计: K=4096 次 fp32 累加, 每次 ULP ~ 1e-7,
    总 rounding error ~ sqrt(K) * 1e-7 * max_value ~ 量级 1e-3.
    """
    torch.manual_seed(42)
    M, N, K = 16, 256, 4096
    x_fp16 = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device=DEVICE, dtype=torch.float16) * 0.1

    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    # 用 fp32 输出避开 fp16 ULP, 才能比较两者的算术误差
    out_int32 = w8a8_scaled_mm_reference(x_q, w_q, sa, sb, out_dtype=torch.float32)
    out_fake = fake_quantize_linear(x_fp16, w_fp16, out_dtype=torch.float32)

    max_diff = (out_int32 - out_fake).abs().max().item()
    absmax = out_int32.abs().max().item()
    rel_err = max_diff / absmax
    assert rel_err < 1e-3, \
        f"fake_quantize 和 int32 reference 偏差太大: rel={rel_err:.2e}"


@requires_cuda
def test_fake_quantize_output_dtype():
    """输出 dtype 必须按参数指定."""
    x = torch.randn(8, 64, device=DEVICE, dtype=torch.float16)
    w = torch.randn(64, 32, device=DEVICE, dtype=torch.float16)
    out_fp16 = fake_quantize_linear(x, w, out_dtype=torch.float16)
    out_fp32 = fake_quantize_linear(x, w, out_dtype=torch.float32)
    assert out_fp16.dtype == torch.float16
    assert out_fp32.dtype == torch.float32


# ============================================================
# 6. cublas_fp16_baseline: 性能对手 (基本契约测试)
# ============================================================

@requires_cuda
def test_cublas_baseline_correctness():
    """cuBLAS baseline 必须和 torch.matmul 直接调用结果一致 (它就是包装)."""
    torch.manual_seed(0)
    x = torch.randn(8, 64, device=DEVICE, dtype=torch.float16)
    w = torch.randn(64, 32, device=DEVICE, dtype=torch.float16)
    out_baseline = cublas_fp16_baseline(x, w)
    out_direct = torch.matmul(x, w)
    assert torch.equal(out_baseline, out_direct)


@requires_cuda
def test_cublas_baseline_with_bias():
    """带 bias 时, 应该加上 bias."""
    torch.manual_seed(0)
    x = torch.randn(8, 64, device=DEVICE, dtype=torch.float16)
    w = torch.randn(64, 32, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(32, device=DEVICE, dtype=torch.float16)
    out = cublas_fp16_baseline(x, w, bias=bias)
    expected = torch.matmul(x, w) + bias
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


# ============================================================
# 7. LLaMA-7B 形状 sanity check
# ============================================================

@requires_cuda
@pytest.mark.parametrize("M,N,K", [
    pytest.param(1, 4096, 4096, id="S1_decode_b1"),
    pytest.param(16, 4096, 4096, id="S2_decode_b16"),
    pytest.param(512, 4096, 4096, id="S3_prefill_512"),
    pytest.param(2048, 4096, 4096, id="S4_prefill_2048"),
])
def test_reference_llama_shapes(M, N, K):
    """LLaMA-7B 4 个标准形状: q/k/v/o projection 的代表."""
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])