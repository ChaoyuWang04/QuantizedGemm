"""
W8A8 scaled matmul 的 PyTorch 金标准实现。

本文件提供整个 pipeline 所需的全部 PyTorch 参考函数,分四组:

  组 1: 量化辅助函数
    - quantize_per_token:    激活 [M,K] -> int8 + scale [M,1]
    - quantize_per_channel:  权重 [K,N] -> int8 + scale [1,N]

  组 2: W8A8 算子参考实现
    - w8a8_scaled_mm_reference:      Layer 1 单测金标准 (int32 累加)
    - w8a8_scaled_mm_reference_fp64: Ultra 金标准 (fp64 累加, 用来验证
                                     int32 版本无溢出)

  组 3: 量化方案探针
    - fake_quantize_linear:  在不写 kernel 的前提下测量化误差本身,
                             用于早期验证量化策略可行性

  组 4: 性能对手 baseline
    - cublas_fp16_baseline:  Triton/CUTLASS kernel 真正要打的对手
                             (W8A8 路线存在的全部理由就是"比 fp16 快")

数学定义 (核心):
    C[i,j] = (sum_k A_q[i,k] * B_q[k,j]) * scale_a[i] * scale_b[j] + bias[j]
"""

from typing import Optional
import torch


# ============================================================
# 组 1: 量化辅助函数
# ============================================================

def quantize_per_token(x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对激活矩阵做对称 per-token 量化。

    每一行 (一个 token 的所有特征) 独立取 absmax 作为该 token 的 scale,
    再映射到 int8。per-token 是激活量化的标准做法,因为不同 token 的数值
    范围差异很大,共享 scale 会让 outlier token 撑大整体 scale,损失精度。

    参数:
        x: fp16/fp32 激活矩阵,[M, K]。
        eps: 防止全零行导致 scale 为 0。

    返回:
        (x_int8, scale_a),其中 scale_a 形状为 [M, 1] 且 dtype 为 fp32。
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

    每一列对应一个输出 channel,独立计算 scale。权重是静态的,可以离线
    把每个输出 channel 的 scale 调到最优。

    参数:
        w: fp16/fp32 权重矩阵,[K, N]。
        eps: 防止全零列导致 scale 为 0。

    返回:
        (w_int8, scale_b),其中 scale_b 形状为 [1, N] 且 dtype 为 fp32。
    """
    w_fp = w.to(torch.float32)
    # 每个输出 channel/列单独求最大绝对值。
    absmax = w_fp.abs().amax(dim=0, keepdim=True).clamp(min=eps)  # [1, N]
    scale_b = absmax / 127.0  # [1, N]
    # 使用对应列的 scale 将权重量化到 int8。
    w_int8 = (w_fp / scale_b).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale_b


# ============================================================
# 组 2: W8A8 算子参考实现
# ============================================================

def w8a8_scaled_mm_reference(
    a: torch.Tensor,           # [M, K] int8
    b: torch.Tensor,           # [K, N] int8
    scale_a: torch.Tensor,     # [M, 1] fp32
    scale_b: torch.Tensor,     # [1, N] fp32
    bias: Optional[torch.Tensor] = None,  # [N]
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    W8A8 scaled matmul 的慢速但严格参考实现 (Layer 1 单测金标准)。

    实现策略: int8 -> int32 累加 -> fp32 dequant -> 目标 dtype。
    这个累加策略和 NVIDIA Tensor Core 的 mma.s8.s8.s32 指令语义一致,
    所以 Triton/CUTLASS kernel 应该和这个版本逐元素对齐。

    参数:
        a: int8 激活矩阵,[M, K]。
        b: int8 权重矩阵,[K, N]。
        scale_a: 激活 per-token scale,[M, 1],fp32。
        scale_b: 权重 per-channel scale,[1, N],fp32。
        bias: 可选 bias,[N],任意 dtype (内部 cast 到 fp32 计算)。
        out_dtype: 输出 dtype,默认 fp16。

    返回:
        形状为 [M, N],dtype 为 out_dtype 的输出矩阵。
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


def w8a8_scaled_mm_reference_fp64(
    a: torch.Tensor,           # [M, K] int8
    b: torch.Tensor,           # [K, N] int8
    scale_a: torch.Tensor,     # [M, 1] fp32
    scale_b: torch.Tensor,     # [1, N] fp32
    bias: Optional[torch.Tensor] = None,  # [N]
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Ultra-reference: 全程用 fp64 累加,不走 int32 路径。

    这个版本存在的唯一目的: 验证 w8a8_scaled_mm_reference 在极端
    形状 (例如 K=16384 或 input 全 127) 下没有 int32 溢出。

    int8 累加的理论安全边界:
        max |x_q * w_q| = 128 * 128 = 16384
        K 个累加: 16384 * K
        int32 上限: 2^31 - 1 ≈ 2.15e9
        => K 安全上限 ≈ 2.15e9 / 16384 ≈ 131072

    所以 K < 131072 时 int32 版本一定不溢出, 但还是用 fp64 版本兜底,
    免得后面 batch matmul 或者 split-K 优化时算错边界。

    用法: 在 test_correctness.py 里加一个测试,
        out_int32 = w8a8_scaled_mm_reference(...)
        out_fp64  = w8a8_scaled_mm_reference_fp64(...)
        assert torch.allclose(out_int32, out_fp64, atol=1e-4)

    参数和返回: 同 w8a8_scaled_mm_reference。
    """
    # 校验同主 reference, 这里精简。
    assert a.dtype == torch.int8 and b.dtype == torch.int8
    assert scale_a.dtype == torch.float32 and scale_b.dtype == torch.float32

    # 直接 cast 到 fp64, 绕开 int32 累加路径。
    a_fp64 = a.to(torch.float64)
    b_fp64 = b.to(torch.float64)
    acc = torch.matmul(a_fp64, b_fp64)  # [M, N] fp64, 数学上和 int32 累加等价

    # scale 也提升到 fp64 做乘法, 误差远低于 fp32 路径。
    out_fp64 = acc * scale_a.double() * scale_b.double()

    if bias is not None:
        out_fp64 = out_fp64 + bias.double()

    return out_fp64.to(out_dtype)


# ============================================================
# 组 3: 量化方案探针 (Layer 2 预备工具)
# ============================================================

def fake_quantize_linear(
    x_fp16: torch.Tensor,      # [M, K] fp16/bf16
    w_fp16: torch.Tensor,      # [K, N] fp16/bf16
    bias: Optional[torch.Tensor] = None,  # [N]
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Fake quantize: 量化 -> 立刻反量化 -> 用 fp16 真算。

    这个函数模拟"完美的 W8A8 实现"的端到端输出:
        x_fp16 --量化--> x_int8 --反量化--> x_dequant_fp16 --matmul--> out

    用途 (Layer 2 预备工具):
        在不写任何 kernel 的前提下, 把模型里所有 nn.Linear 的 forward
        替换成 fake_quantize_linear, 跑 perplexity / MMLU。
        这能告诉你"量化方案本身"在模型上掉点多少。

    它和 w8a8_scaled_mm_reference 的对比:
        ┌─────────────────────────┬──────────────────────────┐
        │ w8a8_reference          │ fake_quantize_linear     │
        ├─────────────────────────┼──────────────────────────┤
        │ 输入: int8              │ 输入: fp16               │
        │ 内部: int32 matmul      │ 内部: fp32 matmul        │
        │ 慢 (CUDA ALU,非 TC)    │ 快 (cuBLAS)              │
        │ 模拟"真 kernel 输出"    │ 模拟"量化误差"           │
        │ 用于 Layer 1 单测       │ 用于 Layer 2 早期验证    │
        └─────────────────────────┴──────────────────────────┘

    数学上, 两者在以下条件下等价 (忽略 fp16 累加 rounding):
        round(X / s_A) * s_A ≈ X
        => 量化-反量化 后的 X 接近原始 X
        => fp32/fp16 matmul 后的结果接近 int8 matmul 后 dequant 的结果

    参数:
        x_fp16: 激活, [M, K]。
        w_fp16: 权重, [K, N]。
        bias: 可选 bias, [N]。
        out_dtype: 输出 dtype。

    返回:
        [M, N] out_dtype 的输出。
    """
    # 量化激活和权重 (复用上面的辅助函数)。
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    # 立刻反量化回 fp32, 注意这里是按 scale 广播相乘恢复连续值。
    # x_dequant ≈ x_fp16 (有量化噪声), w_dequant 同理。
    x_dequant = x_q.float() * sa  # [M, K] fp32
    w_dequant = w_q.float() * sb  # [K, N] fp32

    # 在 fp32 中做 matmul, 数值上比 fp16 matmul 更精确。
    # 这样我们测得的"误差"主要来自量化本身, 而不是 fp16 累加。
    out_fp32 = torch.matmul(x_dequant, w_dequant)

    if bias is not None:
        out_fp32 = out_fp32 + bias.to(torch.float32)

    return out_fp32.to(out_dtype)


# ============================================================
# 组 4: 性能对手 baseline
# ============================================================

def cublas_fp16_baseline(
    x_fp16: torch.Tensor,      # [M, K] fp16/bf16
    w_fp16: torch.Tensor,      # [K, N] fp16/bf16
    bias: Optional[torch.Tensor] = None,  # [N]
) -> torch.Tensor:
    """
    cuBLAS fp16 matmul, 即 W8A8 kernel 真正要打的性能对手。

    Why this baseline:
        W8A8 这条技术路线存在的全部理由是"比 fp16 快"。如果你的
        Triton/CUTLASS W8A8 kernel 比 cuBLAS fp16 matmul 还慢,
        那整个 W8A8 路线在你的硬件上就没意义了 —— 你做的项目就只是
        一个"完整 pipeline 演练", 不是"实际工程价值"。

    实现:
        torch.matmul 在 fp16 输入上会自动调用 cuBLAS gemmEx, 走
        Tensor Core 路径 (sm_80+ 上)。这是 NVIDIA 自家工程师常年
        优化的产物, 是当前能找到的最强 fp16 matmul 实现。

    benchmark 用法:
        # Triton W8A8 latency
        x_q, sa = quantize_per_token(x_fp16)
        w_q, sb = quantize_per_channel(w_fp16)
        t_triton = time(lambda: w8a8_triton(x_q, w_q, sa, sb))

        # cuBLAS fp16 latency (对手)
        t_cublas = time(lambda: cublas_fp16_baseline(x_fp16, w_fp16))

        speedup = t_cublas / t_triton  # > 1 表示赢, < 1 表示输

    在 RTX 4090 上的现实预期:
        Shape S1 (1x4096x4096):   memory-bound, W8A8 难赢 fp16
                                   (反量化开销吃掉收益)
        Shape S2 (16x4096x4096):  memory-bound, 持平或微赢
        Shape S3 (512x4096x4096): balanced, 1.5-2x 加速可期
        Shape S4 (2048x4096x4096):compute-bound, 2-4x 加速可期

    参数:
        x_fp16: 激活, [M, K]。
        w_fp16: 权重, [K, N]。
        bias: 可选 bias, [N]。

    返回:
        [M, N] fp16 的输出。
    """
    # PyTorch 在 fp16 输入下自动 dispatch 到 cuBLAS Tensor Core 路径。
    out = torch.matmul(x_fp16, w_fp16)

    if bias is not None:
        out = out + bias.to(out.dtype)

    return out


# ============================================================
# Smoke test: 把五个函数都跑一遍, 确认无 crash 且数值合理
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    M, N, K = 16, 4096, 4096
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构造模拟激活和权重, 权重乘 0.1 让数值范围更接近真实模型分布。
    x_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1

    # 量化, 准备 W8A8 输入。
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    print("=" * 70)
    print(f"Smoke test: M={M}, N={N}, K={K}, device={device}")
    print("=" * 70)

    # --- 主 reference (int32 累加) ---
    out_int32 = w8a8_scaled_mm_reference(x_q, w_q, sa, sb)
    print(f"\n[1] w8a8_scaled_mm_reference (int32 acc)")
    print(f"    shape={out_int32.shape}, dtype={out_int32.dtype}")
    print(f"    mean={out_int32.mean():.4f}, std={out_int32.std():.4f}, "
          f"absmax={out_int32.abs().max():.4f}")

    # --- ultra reference (fp64 累加), 验证 int32 版本无溢出 ---
    # 注意: fp16 输出版本的 diff 受 fp16 ULP 限制, 不反映溢出。
    # fp16 在 |x|=30 附近 ULP 约 0.03, 所以 fp16 diff 在 1e-2 量级是正常的。
    # 真正能检测溢出的是 fp32 输出路径的 diff。
    out_int32_fp32 = w8a8_scaled_mm_reference(x_q, w_q, sa, sb, out_dtype=torch.float32)
    out_fp64_fp32 = w8a8_scaled_mm_reference_fp64(x_q, w_q, sa, sb, out_dtype=torch.float32)
    diff_fp32_path = (out_int32_fp32 - out_fp64_fp32).abs().max().item()
    print(f"\n[2] w8a8_scaled_mm_reference_fp64 (ultra ref)")
    print(f"    fp32-path diff (int32 vs fp64): {diff_fp32_path:.2e}")
    sentinel = "OK no overflow" if diff_fp32_path < 1e-3 else "WARN possible overflow"
    print(f"    {sentinel}")
    # 留一个 fp16 输出的 diff 给后面对比使用
    diff_int32_vs_fp64 = diff_fp32_path

    # --- fake quantize: 量化策略本身的误差 ---
    out_fake = fake_quantize_linear(x_fp16, w_fp16)
    diff_fake_vs_int32 = (out_fake.float() - out_int32.float()).abs().max().item()
    print(f"\n[3] fake_quantize_linear (量化方案探针)")
    print(f"    max_abs_diff vs int32 version: {diff_fake_vs_int32:.2e}")
    print(f"    含义: int8 实算 vs fp16 模拟量化, 应接近")

    # --- cuBLAS fp16 baseline: 真正性能对手, 用来量化误差 ---
    out_fp16 = cublas_fp16_baseline(x_fp16, w_fp16)
    diff_quant_vs_fp16 = (out_int32.float() - out_fp16.float()).abs().max().item()
    rel_quant_error = diff_quant_vs_fp16 / out_fp16.abs().max().item()
    print(f"\n[4] cublas_fp16_baseline (性能对手 = fp16 真值)")
    print(f"    max_abs_diff (W8A8 vs fp16): {diff_quant_vs_fp16:.4f}")
    print(f"    relative quantization error: {rel_quant_error*100:.2f}%")
    print(f"    含义: 量化本身的精度损失, W8A8 不可能比这更准")

    # --- 一致性总结 ---
    print(f"\n" + "=" * 70)
    print("一致性检查结果:")
    print(f"  int32 acc  vs fp64 acc:  {diff_int32_vs_fp64:.2e}  (< 1e-3 = 安全)")
    print(f"  fake quant vs int32 acc: {diff_fake_vs_int32:.2e}  (< 1e-2 = 量化方案自洽)")
    print(f"  W8A8       vs fp16 true: {rel_quant_error*100:.2f}%   (< 5% = 量化可用)")
    print("=" * 70)