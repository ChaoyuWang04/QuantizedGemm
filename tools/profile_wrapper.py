"""
profile_wrapper.py - 测 wrapper 中每一步的耗时, 定位真正的 overhead 来源.

为什么需要这个:
    诊断脚本告诉我们 wrapper 总开销约 30us, 但具体哪一行慢, 是推测.
    在动手优化前, 先用真实数据定位, 避免改错地方.

方法:
    把 wrapper 拆成几个独立可测的步骤, 每步用 cuda.Event 测.
    跑 1000 次取中位数 (因为每步可能很短, 抖动相对大).

执行:
    python tools/profile_wrapper.py
"""

import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).parent.parent))
from reference.torch_reference import quantize_per_token, quantize_per_channel
from triton_kernel.w8a8_autotuned import _w8a8_scaled_mm_kernel_autotuned


def time_block(name: str, fn, repeat: int = 1000):
    """对一段操作计时, 返回 us 中位数."""
    # warmup
    for _ in range(20):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    median_us = times[len(times) // 2] * 1000
    p95_us = times[int(len(times) * 0.95)] * 1000
    print(f"  {name:<40} median={median_us:>8.2f} us, p95={p95_us:>8.2f} us")
    return median_us


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # S2 形状 (M=16 是 wrapper overhead 最暴露的场景)
    torch.manual_seed(42)
    M, N, K = 16, 4096, 4096
    print(f"Profiling shape S2: M={M}, N={N}, K={K}")
    print()

    x_fp16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1
    x_q, sa = quantize_per_token(x_fp16)
    w_q, sb = quantize_per_channel(w_fp16)

    # 预先触发 autotune (我们要测 wrapper, 不是 autotune)
    print("Warming up autotune cache...")
    from triton_kernel.w8a8_autotuned import w8a8_scaled_mm_triton as w8a8_autotuned
    _ = w8a8_autotuned(x_q, w_q, sa, sb)
    torch.cuda.synchronize()
    print()

    print("=" * 80)
    print("Wrapper 各步骤独立耗时 (单位 us)")
    print("=" * 80)

    # ---- 整个 wrapper ----
    t_full_wrapper = time_block(
        "[FULL] w8a8_scaled_mm_triton (整个 wrapper)",
        lambda: w8a8_autotuned(x_q, w_q, sa, sb),
    )

    print()
    print("--- 分解 wrapper 的每一步 ---")

    # ---- 步骤 1: 8 个 assert ----
    def run_asserts():
        assert x_q.dtype == torch.int8
        assert w_q.dtype == torch.int8
        assert sa.dtype == torch.float32
        assert sb.dtype == torch.float32
        assert x_q.is_cuda and w_q.is_cuda
        assert x_q.shape[1] == w_q.shape[0]
        m, k = x_q.shape
        k2, n = w_q.shape
        assert k == k2

    t_asserts = time_block("Step 1: 8 个 assert", run_asserts)

    # ---- 步骤 2: torch.empty 分配 output ----
    t_empty = time_block(
        "Step 2: torch.empty((M,N), fp16)",
        lambda: torch.empty((M, N), dtype=torch.float16, device="cuda"),
    )

    # ---- 步骤 3: scale_a.squeeze(-1).contiguous() ----
    t_squeeze_sa = time_block(
        "Step 3a: scale_a.squeeze(-1).contiguous()",
        lambda: sa.squeeze(-1).contiguous(),
    )
    t_squeeze_sb = time_block(
        "Step 3b: scale_b.squeeze(0).contiguous()",
        lambda: sb.squeeze(0).contiguous(),
    )

    # ---- 步骤 4: lambda grid + cdiv ----
    def make_grid():
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )
        return grid

    t_grid = time_block("Step 4: 构造 lambda grid", make_grid)

    # ---- 步骤 5: 纯 kernel 调用 (不包含上面任何步骤) ----
    # 预先准备好所有参数, 只测 kernel launch + 执行
    c_pre = torch.empty((M, N), dtype=torch.float16, device="cuda")
    sa_1d = sa.squeeze(-1).contiguous()
    sb_1d = sb.squeeze(0).contiguous()

    # 从 autotune cache 拿 best config
    cache = _w8a8_scaled_mm_kernel_autotuned.cache
    best_config = list(cache.values())[0]
    BLOCK_M = best_config.kwargs["BLOCK_M"]
    BLOCK_N = best_config.kwargs["BLOCK_N"]
    BLOCK_K = best_config.kwargs["BLOCK_K"]
    grid_tuple = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    def run_kernel_only():
        _w8a8_scaled_mm_kernel_autotuned.fn[grid_tuple](
            x_q, w_q, c_pre,
            sa_1d, sb_1d,
            M, N, K,
            x_q.stride(0), x_q.stride(1),
            w_q.stride(0), w_q.stride(1),
            c_pre.stride(0), c_pre.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=best_config.num_warps,
            num_stages=best_config.num_stages,
        )

    t_kernel = time_block(
        "Step 5: 纯 kernel 调用 (不含 wrapper)",
        run_kernel_only,
    )

    # ---- 求和 vs 实测 wrapper ----
    print()
    print("=" * 80)
    print("汇总分析")
    print("=" * 80)
    print()

    sum_of_parts = (t_asserts + t_empty + t_squeeze_sa + t_squeeze_sb +
                    t_grid + t_kernel)
    overhead = t_full_wrapper - t_kernel

    print(f"Sum of profiled parts:  {sum_of_parts:>8.2f} us")
    print(f"Full wrapper (measured): {t_full_wrapper:>8.2f} us")
    print(f"Pure kernel only:        {t_kernel:>8.2f} us")
    print()
    print(f"Wrapper overhead (= full - kernel): {overhead:.2f} us")
    print(f"Kernel utilization in wrapper: {t_kernel/t_full_wrapper*100:.1f}%")
    print()
    print("各部分对 overhead 的贡献:")
    overhead_items = [
        ("assert", t_asserts),
        ("torch.empty", t_empty),
        ("squeeze(sa)", t_squeeze_sa),
        ("squeeze(sb)", t_squeeze_sb),
        ("grid lambda", t_grid),
    ]
    overhead_items.sort(key=lambda x: -x[1])
    for name, val in overhead_items:
        pct = val / overhead * 100 if overhead > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {name:<20} {val:>6.2f} us  ({pct:>5.1f}%) {bar}")

    print()
    print("=" * 80)
    print("结论")
    print("=" * 80)
    print()

    if t_squeeze_sa + t_squeeze_sb > overhead * 0.4:
        print("FINDING: squeeze().contiguous() 是 wrapper overhead 主因.")
        print("  -> Fix: 避免 squeeze, 直接传 [M,1] 和 [1,N] 给 kernel,")
        print("         kernel 内部用 broadcast 即可")
    elif t_asserts > overhead * 0.3:
        print("FINDING: assert 检查占比高.")
        print("  -> Fix: 加 release/debug 模式开关, 生产环境跳过 assert")
    elif t_empty > overhead * 0.3:
        print("FINDING: torch.empty 分配开销大.")
        print("  -> Fix: 让调用方传 output buffer, 复用内存")
    else:
        print("FINDING: overhead 分散, 没有单一主因.")
        print("  -> 可能是 PyTorch/Triton 框架级 launch overhead, 难以优化")


if __name__ == "__main__":
    main()