"""
diagnose_precision.py - 用多指标精度诊断, 验证 W8A8Linear 健康度.

为什么单独写这个:
    smoke test 显示 mean rel diff 5.80%, 看起来"超标"了 (> 5% 警告).
    但 W8A8 的物理精度损失就在 2-6% 范围, 5.80% 是健康的.
    
    用更稳健的指标 (cosine similarity, SNR) 重新评估, 这些指标在
    "输出值有零或近零" 时不会被误差放大.

指标对照:
    mean rel diff:       不稳健 (除零放大)
    max rel diff:        极不稳健 (单点异常)
    cosine similarity:   稳健 (向量整体方向)
    SNR (dB):            稳健 (信号 vs 误差)
    MSE (绝对误差²平均): 稳健 (绝对量)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from integration.w8a8_linear import W8A8Linear


def evaluate_precision(out_fp16, out_w8a8, name=""):
    """多指标精度评估."""
    fp16_f = out_fp16.float().flatten()
    w8a8_f = out_w8a8.float().flatten()
    diff = fp16_f - w8a8_f
    
    # ----- 1. Cosine similarity -----
    # 衡量两个向量的"方向相似度", 不受 magnitude 影响
    # 1.0 = 完美, 0.99+ = 优秀, 0.95+ = 良好
    cos_sim = torch.nn.functional.cosine_similarity(
        fp16_f.unsqueeze(0), w8a8_f.unsqueeze(0)
    ).item()
    
    # ----- 2. SNR (Signal-to-Noise Ratio) in dB -----
    # 信号能量 vs 误差能量, 量化精度的"信噪比"
    # > 30 dB = 优秀, > 20 dB = 良好, > 10 dB = 可接受
    signal_power = (fp16_f ** 2).mean().item()
    noise_power = (diff ** 2).mean().item()
    snr_db = 10 * torch.log10(torch.tensor(signal_power / max(noise_power, 1e-20))).item()
    
    # ----- 3. MSE 和 max abs error -----
    mse = (diff ** 2).mean().item()
    max_abs = diff.abs().max().item()
    
    # ----- 4. Output magnitude statistics (帮助 sanity check) -----
    fp16_abs_mean = fp16_f.abs().mean().item()
    fp16_abs_std = fp16_f.abs().std().item()
    
    # ----- 5. 相对误差但只看 "non-tiny" output (绕开除零) -----
    # 用 fp16 输出的中位数作为下限, 避免极小值放大
    # 注意: 不能用 quantile() — PyTorch 对 > 1600万元素的 tensor 会报错.
    # median() 没大小限制, 等价于 quantile(0.5)
    threshold = fp16_f.abs().median().item()
    mask = fp16_f.abs() > threshold
    if mask.sum() > 0:
        rel_diff_filtered = (diff[mask].abs() / fp16_f[mask].abs()).mean().item()
    else:
        rel_diff_filtered = float('nan')
    
    # ----- 打印 -----
    print(f"\n--- {name} ---")
    print(f"  Output magnitude: mean_abs={fp16_abs_mean:.4f}, std_abs={fp16_abs_std:.4f}")
    print(f"  MSE:                    {mse:.4e}")
    print(f"  Max abs error:          {max_abs:.4e}")
    print(f"  Cosine similarity:      {cos_sim:.6f}  {'PASS' if cos_sim > 0.999 else 'WARN' if cos_sim > 0.99 else 'FAIL'}")
    print(f"  SNR:                    {snr_db:.2f} dB    {'PASS' if snr_db > 25 else 'WARN' if snr_db > 15 else 'FAIL'}")
    print(f"  Filtered rel diff (>p50): {rel_diff_filtered*100:.2f}%   {'PASS' if rel_diff_filtered < 0.03 else 'WARN' if rel_diff_filtered < 0.06 else 'FAIL'}")
    
    return {
        "cos_sim": cos_sim,
        "snr_db": snr_db,
        "mse": mse,
        "rel_diff_filtered": rel_diff_filtered,
    }


def main():
    print("=" * 75)
    print("W8A8Linear Precision Diagnosis (multi-metric)")
    print("=" * 75)
    
    torch.manual_seed(42)
    in_features = 896
    out_features = 896
    
    # 1. 创建 fp16 和 w8a8 版本
    fp16_linear = nn.Linear(in_features, out_features, bias=False).cuda().half()
    w8a8_linear = W8A8Linear(in_features, out_features, bias=False).cuda()
    w8a8_linear.load_from_fp16_linear(fp16_linear)
    
    # 2. 测试场景 1: 标准输入 (batch=2, seq=128, hidden=896)
    print("\n" + "=" * 75)
    print("Test 1: Standard transformer input shape [2, 128, 896]")
    print("=" * 75)
    x1 = torch.randn(2, 128, in_features, device="cuda", dtype=torch.float16) * 0.1
    out_fp16 = fp16_linear(x1)
    out_w8a8 = w8a8_linear(x1)
    metrics1 = evaluate_precision(out_fp16, out_w8a8, "Batch=2, Seq=128")
    
    # 3. 测试场景 2: Decode 单 token [1, 896]
    print("\n" + "=" * 75)
    print("Test 2: Decode single token [1, 896]")
    print("=" * 75)
    x2 = torch.randn(1, in_features, device="cuda", dtype=torch.float16) * 0.1
    out_fp16 = fp16_linear(x2)
    out_w8a8 = w8a8_linear(x2)
    metrics2 = evaluate_precision(out_fp16, out_w8a8, "M=1 Decode")
    
    # 4. 测试场景 3: Prefill 大 batch [16, 2048, 896]
    print("\n" + "=" * 75)
    print("Test 3: Prefill large batch [16, 2048, 896]")
    print("=" * 75)
    x3 = torch.randn(16, 2048, in_features, device="cuda", dtype=torch.float16) * 0.1
    out_fp16 = fp16_linear(x3)
    out_w8a8 = w8a8_linear(x3)
    metrics3 = evaluate_precision(out_fp16, out_w8a8, "Batch=16, Seq=2048")
    
    # 5. 测试场景 4: 真实分布 (不是 *0.1 缩放)
    print("\n" + "=" * 75)
    print("Test 4: Realistic LLM activation distribution")
    print("=" * 75)
    # LLM 中间层激活通常 std ≈ 1.0, 有一些 outlier
    x4 = torch.randn(2, 128, in_features, device="cuda", dtype=torch.float16)
    out_fp16 = fp16_linear(x4)
    out_w8a8 = w8a8_linear(x4)
    metrics4 = evaluate_precision(out_fp16, out_w8a8, "Standard normal input (std=1)")
    
    # 6. 总结
    print()
    print("=" * 75)
    print("总结")
    print("=" * 75)
    print()
    print(f"{'Test':<28} {'CosSim':<12} {'SNR (dB)':<12} {'Filtered RelDiff':<18}")
    print("-" * 75)
    tests = [
        ("Batch=2, Seq=128", metrics1),
        ("M=1 Decode", metrics2),
        ("Batch=16, Seq=2048", metrics3),
        ("Standard normal", metrics4),
    ]
    for name, m in tests:
        print(f"{name:<28} {m['cos_sim']:<12.6f} {m['snr_db']:<12.2f} {m['rel_diff_filtered']*100:<17.2f}%")
    
    print()
    print("结论:")
    avg_cos = sum(m['cos_sim'] for _, m in tests) / len(tests)
    avg_snr = sum(m['snr_db'] for _, m in tests) / len(tests)
    print(f"  Avg cosine similarity: {avg_cos:.6f}")
    print(f"  Avg SNR:               {avg_snr:.2f} dB")
    print()
    if avg_cos > 0.999 and avg_snr > 25:
        print("PASS: W8A8Linear 精度健康, 可以进入下一步集成")
    elif avg_cos > 0.99 and avg_snr > 15:
        print("OK: 精度可接受, 进入下一步; 集成时注意监控整网 perplexity")
    else:
        print("WARN: 精度异常, 建议检查量化逻辑")


if __name__ == "__main__":
    main()