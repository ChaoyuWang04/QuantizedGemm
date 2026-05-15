"""
W8A8Linear - nn.Module wrapper around our Triton W8A8 kernel.

替换流程:
    # 原模型
    nn.Linear(in_features=896, out_features=896, bias=False)
    
    # 替换后
    W8A8Linear(in_features=896, out_features=896, bias=False)
    
    # 两者 forward 接口完全一致, 调用方代码不用改

权重布局:
    原 nn.Linear.weight:  [out_features, in_features]  fp16
    W8A8Linear.weight_int8: [in_features, out_features]  int8   ← 已 transpose
    W8A8Linear.weight_scale: [1, out_features]           fp32   ← per-channel
    
    Transpose 在 量化时一次性做掉, forward 不再转.

精度:
    激活: per-token 动态量化 (每次 forward)
    权重: per-channel 离线量化 (一次永久)
    
    这是 W8A8 标准配置, 平衡精度和性能.
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from triton_kernel.w8a8_autotuned import w8a8_scaled_mm_triton


class W8A8Linear(nn.Module):
    """
    替代 nn.Linear, 内部用 W8A8 Triton kernel.
    
    接口完全兼容 nn.Linear:
        - 输入: [..., in_features] fp16
        - 输出: [..., out_features] fp16
        - 支持任意 batch 维度 (会 flatten 后调 kernel, 再 reshape 回来)
    """
    
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, device=None, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        
        # ----- 权重: int8 + per-channel scale -----
        # weight_int8 shape: [in_features, out_features]  (已 transpose, 直接给 kernel)
        # weight_scale shape: [1, out_features]            (per-channel scale)
        # 用 register_buffer 而不是 Parameter, 因为我们不训练这些
        self.register_buffer(
            "weight_int8",
            torch.zeros((in_features, out_features), dtype=torch.int8, device=device),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones((1, out_features), dtype=torch.float32, device=device),
        )
        
        # ----- bias: fp16 (如果有) -----
        if bias:
            # 用 Parameter 是为了和 nn.Linear 接口一致 (虽然我们不训练)
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=dtype, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播.
        
        Args:
            x: [..., in_features] fp16
        Returns:
            [..., out_features] fp16
        """
        # ----- 处理任意 batch 维度 -----
        # 比如 [batch=2, seq=128, hidden=896] -> 先 reshape 成 [256, 896]
        # 调完 kernel 后 -> reshape 回 [2, 128, out_features]
        orig_shape = x.shape
        x_2d = x.view(-1, self.in_features)   # [M, in_features]
        M = x_2d.shape[0]
        
        # ----- 量化激活 (per-token) -----
        # 每个 token 一个 scale: x_int8 [M, in_features], x_scale [M, 1]
        x_int8, x_scale = _quantize_per_token(x_2d)
        
        # ----- 调 Triton W8A8 kernel -----
        # 输出 [M, out_features] fp16
        out_2d = w8a8_scaled_mm_triton(
            x_int8,                      # [M, in_features] int8
            self.weight_int8,            # [in_features, out_features] int8
            x_scale,                     # [M, 1] fp32
            self.weight_scale,           # [1, out_features] fp32
        )
        
        # ----- 加 bias (如果有) -----
        if self.bias is not None:
            out_2d = out_2d + self.bias
        
        # ----- reshape 回原 batch 维度 -----
        out_shape = orig_shape[:-1] + (self.out_features,)
        return out_2d.view(out_shape)
    
    def extra_repr(self) -> str:
        """Make print(model) show useful info."""
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.has_bias}, dtype=int8")
    
    @torch.no_grad()
    def load_from_fp16_linear(self, fp16_linear: nn.Linear):
        """
        从一个 fp16 nn.Linear 加载权重, 量化为 int8.
        
        这是离线量化的入口:
            fp16_linear.weight: [out_features, in_features] fp16
            ↓ transpose + per-channel quantize
            self.weight_int8: [in_features, out_features] int8
            self.weight_scale: [1, out_features] fp32
        """
        assert fp16_linear.in_features == self.in_features
        assert fp16_linear.out_features == self.out_features
        
        # 1. Transpose fp16 weight: [out, in] -> [in, out]
        w_fp16 = fp16_linear.weight.data.t().contiguous().to(torch.float16)
        
        # 2. Per-channel 量化 (每个输出通道一个 scale)
        # scale = max(|w|) per column / 127
        # int8 = round(w / scale)
        w_int8, w_scale = _quantize_per_channel(w_fp16)
        
        # 3. 写入 buffer
        self.weight_int8.copy_(w_int8)
        self.weight_scale.copy_(w_scale)
        
        # 4. Bias (如果都有)
        if fp16_linear.bias is not None:
            assert self.bias is not None, "Source has bias but target doesn't"
            self.bias.data.copy_(fp16_linear.bias.data.to(torch.float16))


# ============================================================
# 量化辅助函数 (从 reference 借鉴, 但要可独立使用)
# ============================================================

@torch.no_grad()
def _quantize_per_token(x_fp16: torch.Tensor):
    """
    Per-token 动态量化激活.
    
    Args:
        x_fp16: [M, K] fp16
    Returns:
        x_int8: [M, K] int8
        scale: [M, 1] fp32
    """
    # 每行的绝对值最大值
    abs_max = x_fp16.abs().amax(dim=-1, keepdim=True).to(torch.float32)
    # 防止除零 (输入全 0 的极端情况)
    abs_max = abs_max.clamp(min=1e-8)
    # scale = max / 127 (int8 范围 [-128, 127], 但用 127 避免溢出)
    scale = abs_max / 127.0
    # 量化
    x_int8 = (x_fp16.to(torch.float32) / scale).round().clamp(-128, 127).to(torch.int8)
    return x_int8, scale


@torch.no_grad()
def _quantize_per_channel(w_fp16: torch.Tensor):
    """
    Per-channel 离线量化权重.
    
    Args:
        w_fp16: [K, N] fp16  (已 transpose, K=in_features, N=out_features)
    Returns:
        w_int8: [K, N] int8
        scale: [1, N] fp32
    """
    # 每列 (输出通道) 的绝对值最大值
    abs_max = w_fp16.abs().amax(dim=0, keepdim=True).to(torch.float32)   # [1, N]
    abs_max = abs_max.clamp(min=1e-8)
    scale = abs_max / 127.0   # [1, N]
    w_int8 = (w_fp16.to(torch.float32) / scale).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("W8A8Linear -- Smoke Test")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # 模拟 Qwen2.5-0.5B 的一个 q_proj: hidden=896
    in_features = 896
    out_features = 896
    
    # 1. 创建 fp16 reference
    fp16_linear = nn.Linear(in_features, out_features, bias=False).cuda().half()
    
    # 2. 创建 W8A8 版本并从 fp16 加载
    w8a8_linear = W8A8Linear(in_features, out_features, bias=False).cuda()
    w8a8_linear.load_from_fp16_linear(fp16_linear)
    
    print(f"\nfp16_linear: {fp16_linear}")
    print(f"w8a8_linear: {w8a8_linear}")
    print()
    
    # 3. 前向, 测精度对齐
    # 输入: [batch=2, seq=128, hidden=896] 模拟真实模型场景
    x = torch.randn(2, 128, in_features, device="cuda", dtype=torch.float16) * 0.1
    
    out_fp16 = fp16_linear(x)
    out_w8a8 = w8a8_linear(x)
    
    print(f"Input shape: {x.shape}, dtype: {x.dtype}")
    print(f"fp16 output: shape={out_fp16.shape}, dtype={out_fp16.dtype}")
    print(f"w8a8 output: shape={out_w8a8.shape}, dtype={out_w8a8.dtype}")
    
    # 4. 精度对比
    abs_diff = (out_fp16.float() - out_w8a8.float()).abs()
    rel_diff = abs_diff / (out_fp16.float().abs() + 1e-6)
    
    print(f"\n--- Precision vs fp16 ---")
    print(f"max abs diff:       {abs_diff.max().item():.4e}")
    print(f"mean abs diff:      {abs_diff.mean().item():.4e}")
    print(f"max rel diff:       {rel_diff.max().item():.4f}")
    print(f"mean rel diff:      {rel_diff.mean().item():.4f}")
    
    # W8A8 vs fp16 应该有 1-3% 的精度损失 (这是量化的物理代价, 不是 bug)
    mean_rel = rel_diff.mean().item()
    if mean_rel < 0.05:
        print(f"PASS — mean rel diff {mean_rel*100:.2f}% < 5%")
    else:
        print(f"WARN — mean rel diff {mean_rel*100:.2f}% > 5%, 检查量化逻辑")
    
    # 5. 测 1D 输入 (单个 token, 模拟 decode 场景)
    print("\n--- Test 1D input (decode batch=1) ---")
    x_1d = torch.randn(1, in_features, device="cuda", dtype=torch.float16) * 0.1
    out_fp16_1d = fp16_linear(x_1d)
    out_w8a8_1d = w8a8_linear(x_1d)
    rel_diff_1d = ((out_fp16_1d.float() - out_w8a8_1d.float()).abs() /
                   (out_fp16_1d.float().abs() + 1e-6)).mean().item()
    print(f"M=1 mean rel diff: {rel_diff_1d*100:.2f}%")
    
    # 6. 测 has bias
    print("\n--- Test with bias ---")
    fp16_linear_b = nn.Linear(in_features, out_features, bias=True).cuda().half()
    w8a8_linear_b = W8A8Linear(in_features, out_features, bias=True).cuda()
    w8a8_linear_b.load_from_fp16_linear(fp16_linear_b)
    
    out_b_fp16 = fp16_linear_b(x)
    out_b_w8a8 = w8a8_linear_b(x)
    rel_diff_b = ((out_b_fp16.float() - out_b_w8a8.float()).abs() /
                  (out_b_fp16.float().abs() + 1e-6)).mean().item()
    print(f"With bias mean rel diff: {rel_diff_b*100:.2f}%")
    
    print()
    print("=" * 70)
    print("Smoke test 完成")
    print("=" * 70)