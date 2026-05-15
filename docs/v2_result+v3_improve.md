INT8 TOPS (dense):   838
FP16 TFLOPS:         419
HBM Bandwidth:       1792 GB/s
AI Crossover:        467.6 ops/byte
Warmup per bench:    0.5s

Global GPU pre-warmup (5.0s sustained workload)...
  -> ran 49623 iterations, GPU should be at boost clock now

Note: 第一次跑 autotuned 版本会触发 autotune (每形状 20-40s)

Running S1 (M=1, N=4096, K=4096) ... done (3.5s)
Running S2 (M=16, N=4096, K=4096) ... done (3.2s)
Running S3 (M=512, N=4096, K=4096) ... done (3.5s)
Running S4 (M=2048, N=4096, K=4096) ... done (4.1s)

============================================================================================
Performance Benchmark - W8A8 Triton vs cuBLAS fp16 (latency in microseconds)
============================================================================================

Shape (M,N,K)               naive      autotuned   cuBLAS fp16  autotuned TFLOPS  Roof%   Bound
--------------------------------------------------------------------------------------------
S1    (   1, 4096, 4096)    111.01     47.65       49.41        0.70              19.7%   memory
S2    (  16, 4096, 4096)    48.64      49.12       28.19        10.93             19.3%   memory
S3    ( 512, 4096, 4096)    105.02     82.08       105.50       209.31            25.0%   compute
S4    (2048, 4096, 4096)    280.54     189.15      367.68       363.30            43.4%   compute

============================================================================================
Speedup Analysis (基准: Triton autotuned, 数字>1 表示 autotuned 更快)
============================================================================================

Shape naive -> autotuned (优化进度)               autotuned vs cuBLAS fp16
--------------------------------------------------------------------------------------------
S1    2.33x faster than naive                 1.04x faster than cuBLAS
S2    0.99x slower than naive                 0.57x slower than cuBLAS
S3    1.28x faster than naive                 1.29x faster than cuBLAS
S4    1.48x faster than naive                 1.94x faster than cuBLAS

============================================================================================
解读
============================================================================================

对手身份:
  naive (v1)     = 我们第一版, 固定 BLOCK=64, 当 baseline
  autotuned (v2) = 我们当前最优版, autotune 自动选 BLOCK
  cuBLAS fp16    = torch.matmul fp16, 用户的'替代方案'

怎么读:
  naive -> autotuned > 1x: 优化有效 (autotune 比固定 BLOCK 快)
  autotuned vs cuBLAS > 1x: 用 W8A8 比 fp16 快, 路线有价值
  autotuned vs cuBLAS < 1x: 在该形状上 W8A8 不如 fp16, 路线无意义

  Roof%: 距 Roofline 物理上限的百分比
    > 70%: 接近极限, 优化空间小
    30-70%: 健康范围
    < 30%: 还有大量优化空间

进入 v3 优化大形状:加 swizzle/super-grouping, 让 S4 从 43% Roof% 推到 60%+