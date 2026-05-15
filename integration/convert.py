"""
convert.py - 把 HF 模型中的所有 nn.Linear 替换为 W8A8Linear.

设计决策:
    1. In-place 替换: 直接修改传入的 model 对象
    2. 默认跳过 lm_head: 量化它会破坏 softmax 输出 (业界共识)
    3. 强制跳过 embed_tokens / tied weights
    4. 检查 in/out features 必须能被 16 整除 (kernel BLOCK 限制)
    5. 宽松模式: 失败的 Linear 跳过, 最后报告
    6. 进度可见: 替换时打 log, 让用户知道改了多少

API:
    convert_model_to_w8a8(model, skip_patterns=["lm_head"]) -> stats dict

执行 (作为脚本验证):
    python integration/convert.py
"""

import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from integration.w8a8_linear import W8A8Linear


# ============================================================
# 默认 skip 规则
# ============================================================

# 名字里含这些子串的 Linear 不替换 (按子串匹配 module name)
DEFAULT_SKIP_PATTERNS = [
    "lm_head",          # 输出层, softmax 敏感, 业界共识跳过
    "embed",            # embedding 不是 nn.Linear, 但保险起见
]

# 必须满足的形状条件: in/out features 能被这个数整除
# 16 是因为我们 kernel 最小 BLOCK 是 16
MIN_FEATURE_MULTIPLE = 16


# ============================================================
# 工具: 按 module name 替换
# ============================================================

def _get_parent_module(model: nn.Module, full_name: str):
    """
    给定 'model.layers.0.self_attn.q_proj', 返回 (parent_module, 'q_proj').
    
    用于 setattr 替换. 不能直接用 model.layers.0.self_attn.q_proj = ...,
    需要先拿到 self_attn, 再 setattr(parent, 'q_proj', ...).
    """
    parts = full_name.split(".")
    parent = model
    for part in parts[:-1]:
        # 数字索引 (列表 ModuleList) vs 属性名
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def _should_skip(name: str, skip_patterns: List[str]) -> Optional[str]:
    """
    判断一个 Linear 是否应该跳过. 返回 None = 不跳过, 否则返回跳过原因.
    """
    for pattern in skip_patterns:
        if pattern in name:
            return f"matches skip pattern '{pattern}'"
    return None


def _check_shape_compatible(linear: nn.Linear) -> Optional[str]:
    """
    检查 Linear 形状是否兼容 kernel. 返回 None = 兼容, 否则返回拒绝原因.
    """
    if linear.in_features % MIN_FEATURE_MULTIPLE != 0:
        return (f"in_features={linear.in_features} not divisible by "
                f"{MIN_FEATURE_MULTIPLE}")
    if linear.out_features % MIN_FEATURE_MULTIPLE != 0:
        return (f"out_features={linear.out_features} not divisible by "
                f"{MIN_FEATURE_MULTIPLE}")
    return None


# ============================================================
# 主转换函数
# ============================================================

@torch.no_grad()
def convert_model_to_w8a8(
    model: nn.Module,
    skip_patterns: Optional[List[str]] = None,
    verbose: bool = True,
) -> dict:
    """
    In-place 把 model 中所有合格的 nn.Linear 替换成 W8A8Linear.
    
    Args:
        model: HF transformers 模型 (或任何 nn.Module 子树)
        skip_patterns: 名字包含这些子串的 Linear 不替换. 默认含 'lm_head', 'embed'.
        verbose: 打印每个替换的进度
    
    Returns:
        stats dict: {
            'converted': [(name, in, out), ...],
            'skipped': [(name, reason), ...],
            'failed': [(name, error), ...],
            'total_params_saved': int,  # int8 vs fp16 节省的字节数
        }
    
    Note:
        转换后, model 已经被原地修改. 不返回新 model.
    """
    if skip_patterns is None:
        skip_patterns = DEFAULT_SKIP_PATTERNS

    stats = {
        "converted": [],
        "skipped": [],
        "failed": [],
        "total_params_saved": 0,
    }

    # 1. 先收集所有要替换的 Linear (不要在遍历中替换, 会改变迭代器)
    linears_to_check = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linears_to_check.append((name, module))

    if verbose:
        print(f"Found {len(linears_to_check)} nn.Linear layers in model")
        print(f"Skip patterns: {skip_patterns}")
        print()

    # 2. 逐个处理
    t_start = time.time()
    for name, linear in linears_to_check:
        # 2a. 检查是否在 skip 名单
        skip_reason = _should_skip(name, skip_patterns)
        if skip_reason is not None:
            stats["skipped"].append((name, skip_reason))
            if verbose:
                print(f"  SKIP  {name:<60} ({skip_reason})")
            continue

        # 2b. 检查形状
        shape_issue = _check_shape_compatible(linear)
        if shape_issue is not None:
            stats["skipped"].append((name, shape_issue))
            if verbose:
                print(f"  SKIP  {name:<60} ({shape_issue})")
            continue

        # 2c. 实际替换
        try:
            # 创建 W8A8Linear 同样配置
            w8a8 = W8A8Linear(
                in_features=linear.in_features,
                out_features=linear.out_features,
                bias=(linear.bias is not None),
                device=linear.weight.device,
            )
            # 从 fp16 linear 加载并量化
            w8a8.load_from_fp16_linear(linear)

            # 找到父模块, setattr 替换
            parent, attr_name = _get_parent_module(model, name)
            setattr(parent, attr_name, w8a8)

            # 统计
            saved_bytes = (linear.in_features * linear.out_features *
                          (2 - 1))   # fp16 = 2 bytes, int8 = 1 byte
            stats["converted"].append((name, linear.in_features, linear.out_features))
            stats["total_params_saved"] += saved_bytes

            if verbose:
                print(f"  OK    {name:<60} "
                      f"[{linear.in_features:>5} -> {linear.out_features:>5}]"
                      f"{' +bias' if linear.bias is not None else ''}")
        except Exception as e:
            stats["failed"].append((name, str(e)))
            if verbose:
                print(f"  FAIL  {name:<60} ({e})")

    t_elapsed = time.time() - t_start

    # 3. 报告
    if verbose:
        print()
        print("=" * 80)
        print(f"Conversion Summary (took {t_elapsed:.1f}s)")
        print("=" * 80)
        print(f"  Converted: {len(stats['converted'])} layers")
        print(f"  Skipped:   {len(stats['skipped'])} layers")
        print(f"  Failed:    {len(stats['failed'])} layers")
        print(f"  Memory saved: {stats['total_params_saved']/1e6:.1f} MB "
              f"(fp16->int8 weight)")
        print()
        
        # 显示按 skip 理由分组
        if stats["skipped"]:
            print("Skip reasons:")
            reason_counts = {}
            for name, reason in stats["skipped"]:
                # 取理由的"种类"忽略具体数字
                kind = reason.split(" not divisible")[0] if "divisible" in reason else reason
                reason_counts[kind] = reason_counts.get(kind, 0) + 1
            for kind, count in reason_counts.items():
                print(f"  [{count:>3}] {kind}")
            print()

    return stats


# ============================================================
# 验证用 smoke test (用一个假模型, 不依赖 transformers)
# ============================================================

def _make_fake_qwen_like_model():
    """构造一个 mini Qwen-like 结构, 用于测 convert 而不依赖 HF download."""
    class FakeAttention(nn.Module):
        def __init__(self, hidden=896, num_heads=14, num_kv_heads=2):
            super().__init__()
            head_dim = hidden // num_heads
            self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=True)
            self.k_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=True)
            self.v_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=True)
            self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)
    
    class FakeMLP(nn.Module):
        def __init__(self, hidden=896, intermediate=4864):
            super().__init__()
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)
    
    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeAttention()
            self.mlp = FakeMLP()
    
    class FakeModel(nn.Module):
        def __init__(self, n_layers=24):
            super().__init__()
            self.embed_tokens = nn.Embedding(151936, 896)
            self.layers = nn.ModuleList([FakeLayer() for _ in range(n_layers)])
            self.lm_head = nn.Linear(896, 151936, bias=False)
    
    return FakeModel()


def main():
    print("=" * 80)
    print("convert.py -- Smoke Test on fake Qwen-like model")
    print("=" * 80)
    print()

    # 构造假模型
    model = _make_fake_qwen_like_model().cuda().half()
    
    # Count parameters before
    n_params_before = sum(p.numel() for p in model.parameters())
    print(f"Model params before:  {n_params_before/1e6:.1f}M")
    print()

    # 转换
    stats = convert_model_to_w8a8(model, verbose=True)
    
    # Verify
    print("=" * 80)
    print("Post-conversion verification")
    print("=" * 80)
    
    # 1. 检查所有 Linear 都被替换 (除了 skip)
    remaining_linears = []
    w8a8_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            remaining_linears.append(name)
        if isinstance(module, W8A8Linear):
            w8a8_count += 1
    
    print(f"\n  Remaining nn.Linear: {len(remaining_linears)}")
    print(f"  Now W8A8Linear:      {w8a8_count}")
    
    expected_remaining = sum(1 for name, _ in stats["skipped"])   # 应该有 skip 的还在
    expected_w8a8 = len(stats["converted"])
    
    if len(remaining_linears) == expected_remaining and w8a8_count == expected_w8a8:
        print("  PASS: counts match")
    else:
        print(f"  FAIL: expected {expected_remaining} nn.Linear remaining, {expected_w8a8} W8A8")
    
    # 2. 跑一个假 forward, 验证替换后能 forward
    print("\nQuick forward test (fake layer with W8A8)...")
    layer_0 = model.layers[0]
    x = torch.randn(2, 32, 896, device="cuda", dtype=torch.float16) * 0.1
    
    # Attention
    q = layer_0.self_attn.q_proj(x)
    k = layer_0.self_attn.k_proj(x)
    v = layer_0.self_attn.v_proj(x)
    print(f"  q_proj output: shape={q.shape}, dtype={q.dtype}")
    print(f"  k_proj output: shape={k.shape}, dtype={k.dtype}")
    print(f"  v_proj output: shape={v.shape}, dtype={v.dtype}")
    
    # MLP
    gate = layer_0.mlp.gate_proj(x)
    up = layer_0.mlp.up_proj(x)
    print(f"  gate_proj output: shape={gate.shape}, dtype={gate.dtype}")
    print(f"  up_proj output:   shape={up.shape}, dtype={up.dtype}")
    
    print("\n  PASS: forward through replaced layers works")
    print()


if __name__ == "__main__":
    main()