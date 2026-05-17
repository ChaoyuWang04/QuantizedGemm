✅ CODEX_HOME=/home/chaoyu/projects/tools/.codex
chaoyu@computeinstance-e00zhttcbjhxh7xsrm:~/projects/QuantizedGemm$ nsys profile -o bench_l40s --force-overwrite t
    uv run benchmarks/bench_kernel.py
WARNING: CPU IP/backtrace sampling not supported, disabling.
Try the 'nsys status --environment' command to learn more.

WARNING: CPU context switch tracing not supported, disabling.
Try the 'nsys status --environment' command to learn more.

Collecting data...
GPU:                 NVIDIA L40S
INT8 TOPS (dense):   362
HBM Bandwidth:       864 GB/s
Warmup per bench:    0.5s

Global GPU pre-warmup (5.0s sustained workload)...
  -> ran 58038 iterations, GPU should be at boost clock now

Note: 第一次跑 autotuned/swizzled 会触发 autotune (每形状 20-60s)

Running S1 (M=1, N=4096, K=4096) ... done (15.0s)
Running S2 (M=16, N=4096, K=4096) ... done (15.3s)
Running S3 (M=512, N=4096, K=4096) ... done (6.1s)
Running S4 (M=2048, N=4096, K=4096) ... done (7.0s)

==================================================================================================================
Performance Benchmark - W8A8 Triton (v1/v2.1/v3) vs cuBLAS fp16 (latency in us)
==================================================================================================================

Shape (M,N,K)               naive     autotuned   swizzled  cuBLAS    v3 TFLOPS   v3 Roof%  Bound
---------------------------------------------------------------------------------------------------------------------------------------
S1    (   1, 4096, 4096)    90.69     92.58       95.62     75.49     0.35        20.3%     memory
S2    (  16, 4096, 4096)    89.63     95.10       94.66     51.39     5.67        20.8%     memory
S3    ( 512, 4096, 4096)    155.81    123.74      126.30    107.49    136.02      37.6%     compute
S4    (2048, 4096, 4096)    406.78    232.74      233.50    369.31    294.30      81.3%     compute

=======================================================================================================================================
Speedup Analysis (基准: v3 swizzled, > 1x 表示 swizzled 更快)
=======================================================================================================================================

Shape naive -> swizzled             autotuned -> swizzled (swizzle 收益)      swizzled vs cuBLAS
---------------------------------------------------------------------------------------------------------------------------------------
S1    0.95x slower                  0.97x slower (swizzle 失效)               0.79x slower
S2    0.95x slower                  1.00x (essentially same)                0.54x slower
S3    1.23x faster                  0.98x slower (swizzle 失效)               0.85x slower
S4    1.74x faster                  1.00x (essentially same)                1.58x faster

=======================================================================================================================================
解读
=======================================================================================================================================

对手身份:
  naive (v1)     = 固定 BLOCK=64, baseline
  autotuned (v2.1) = autotune 选 BLOCK 配置
  swizzled (v3)  = autotune + GROUP_M swizzle 提升 L2 命中
  cuBLAS fp16    = 替代方案

怎么读 swizzle 收益:
  autotuned -> swizzled > 1.0x: swizzle 有效 (主要看 S3/S4)
  autotuned -> swizzled ≈ 1.0x: swizzle 无效 (小形状常见, autotune 选 GROUP_M=1)
  autotuned -> swizzled < 1.0x: swizzle 反效果 (不应出现, 因为 GROUP_M=1 是 fallback)
Generating '/tmp/nsys-report-6691.qdstrm'
[1/1] [========================100%] bench_l40s.nsys-rep
Generated:
        /home/chaoyu/projects/QuantizedGemm/bench_l40s.nsys-rep