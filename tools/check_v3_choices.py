"""
check_v3_choices.py - 看 v3 swizzled 在每个形状上 autotune 选了什么.

回答的问题:
    v3 数据和 v2.1 几乎一样, 是因为 autotune 全选 GROUP_M=1 退化了?
    还是因为 swizzle 确实没收益?

执行:
    python tools/check_v3_choices.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from reference.torch_reference import quantize_per_token, quantize_per_channel
from triton_kernel.w8a8_swizzled import (
    _w8a8_scaled_mm_kernel_swizzled,
    w8a8_scaled_mm_triton,
)


SHAPES = {
    "S1": (1, 4096, 4096),
    "S2": (16, 4096, 4096),
    "S3": (512, 4096, 4096),
    "S4": (2048, 4096, 4096),
}


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    print("触发 autotune for each shape...")
    print()

    for shape_id, (M, N, K) in SHAPES.items():
        torch.manual_seed(42)
        x_fp16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
        w_fp16 = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1
        x_q, sa = quantize_per_token(x_fp16)
        w_q, sb = quantize_per_channel(w_fp16)

        # 触发 autotune
        _ = w8a8_scaled_mm_triton(x_q, w_q, sa, sb)
        torch.cuda.synchronize()
        print(f"  {shape_id} (M={M}) autotune done")

    # 从 cache 读 best configs
    cache = _w8a8_scaled_mm_kernel_swizzled.cache

    print()
    print("=" * 90)
    print("Best config selected by autotune for each shape")
    print("=" * 90)
    print()
    print(f"{'Shape':<8}{'M':<6}{'BLOCK_M':<10}{'BLOCK_N':<10}{'BLOCK_K':<10}{'GROUP_M':<10}{'warps':<8}{'stages':<8}")
    print("-" * 90)

    for key, best_config in cache.items():
        # key is a tuple, first 3 elements are M, N, K
        m_key = key[0]
        shape_id = next((sid for sid, (M, N, K) in SHAPES.items() if M == m_key), "?")
        bm = best_config.kwargs.get("BLOCK_M", "?")
        bn = best_config.kwargs.get("BLOCK_N", "?")
        bk = best_config.kwargs.get("BLOCK_K", "?")
        gm = best_config.kwargs.get("GROUP_M", "?")
        nw = best_config.num_warps
        ns = best_config.num_stages
        
        marker = " <-- GROUP_M=1, swizzle 被退化" if gm == 1 else " <-- GROUP_M>1, swizzle 启用"
        print(f"{shape_id:<8}{m_key:<6}{bm:<10}{bn:<10}{bk:<10}{gm:<10}{nw:<8}{ns:<8}{marker}")

    print()
    print("=" * 90)
    print("诊断")
    print("=" * 90)
    print()

    n_active = 0
    n_disabled = 0
    for key, best_config in cache.items():
        gm = best_config.kwargs.get("GROUP_M", 1)
        if gm == 1:
            n_disabled += 1
        else:
            n_active += 1

    print(f"启用 swizzle 的形状数: {n_active}")
    print(f"退化为 GROUP_M=1 的形状数: {n_disabled}")
    print()

    if n_disabled == len(cache):
        print("DIAGNOSIS: 所有形状都退化为 GROUP_M=1, swizzle 完全没被启用")
        print("  -> v3 = v2.1, 没有真正测试 swizzle 效果")
        print("  -> 可能原因:")
        print("     a. autotune 选 GROUP_M=1 时 kernel 比 GROUP_M>1 快几 us")
        print("        (可能 swizzle 公式的额外计算开销 > L2 收益)")
        print("     b. 5090 的 L2 本来就够大, swizzle 无收益")
    elif n_active == len(cache):
        print("DIAGNOSIS: 所有形状都用了 swizzle, 但 benchmark 数据没改善")
        print("  -> swizzle 计算正确, 但 5090 上没 L2 收益")
    else:
        print(f"DIAGNOSIS: 部分形状用 swizzle ({n_active}/{len(cache)})")
        print("  -> 看选了 swizzle 的形状有没有改善")


if __name__ == "__main__":
    main()