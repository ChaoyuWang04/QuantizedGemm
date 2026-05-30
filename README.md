# W8A8 INT8 GEMM Pipeline — 项目档案

> 一个从零实现 INT8 量化矩阵乘法、并集成进真实 LLM 的完整工程项目。
> 时间跨度：2026-04 ~ 2026-05｜硬件：RTX 5090 (Blackwell) / L40S (Ada)｜框架：Triton + PyTorch

---

## 0. 一句话概括

我用 Triton 从零写了一个 W8A8（INT8 权重 + INT8 激活）GEMM kernel，做了完整的
**正确性验证 → 性能优化 → 真实模型集成 → 深度 profiling** 全流程，最终在 Qwen3-0.6B 上
达到 **PPL 1.032x**（接近 AWQ/GPTQ 水平），并在 S4 形状上跑赢 cuBLAS fp16 **2.17x**。

这个项目真正的收获不是某个性能数字，而是**建立了一套"算子工程师"的完整工作流和方法论**。

---

## 1. 为什么做这个

目标是冲 Frontier AI Lab 的 **AI Infra 岗位**。这类岗位的核心能力是：
- 能独立写、调、优化 GPU kernel
- 理解量化推理的端到端链路（不只是 kernel，还有集成、精度、性能）
- 有系统性的性能分析方法论

W8A8 GEMM 是 LLM 推理里最核心的算子之一（vLLM / TensorRT-LLM 的关键路径），
适合作为"一个能讲完整故事"的项目。

**关键决策**：选 **Triton** 而非 CUDA C++ + pybind。
- Trade-off：放弃 5-15% 极致性能，换取 3-5x 开发速度
- 理由：vLLM / SGLang / DeepSeek 都在转向 Triton-first，这是当下生态主流
- 这是一个**主动的、有理由的工程取舍**，不是"不会 CUDA"

---

## 2. 技术方案

### 量化 scheme
| 对象 | 量化粒度 | scale 形状 | 时机 |
|---|---|---|---|
| 激活 (activation) | per-token | `[M, 1]` | forward 时**动态**量化 |
| 权重 (weight) | per-channel | `[1, N]` | **离线**量化一次，永久存 int8 |
| 输出 (output) | fp16 | — | kernel 内 dequant 后输出 |

**为什么非对称**：激活每次输入都变，per-token 动态算开销可接受；权重固定，
per-channel 离线算一次，精度比 per-tensor 好。

### 测试形状（对标 LLaMA-7B）
| 形状 | (M, N, K) | 含义 | Bound |
|---|---|---|---|
| S1 | (1, 4096, 4096) | decode batch=1 | memory |
| S2 | (16, 4096, 4096) | batched decode | memory |
| S3 | (512, 4096, 4096) | 小 prefill | compute |
| S4 | (2048, 4096, 4096) | 大 prefill | compute |

---

## 3. 我做了什么（P0 → P5）

### P0 — 项目结构 + 需求拆解
定义模块边界：reference / kernel / tests / benchmarks / integration / tools / experiments。

### P1 — PyTorch reference + 测试
写了 5 个参考函数（`quantize_per_token`、`quantize_per_channel`、
`w8a8_scaled_mm_reference`、`cublas_fp16_baseline`、`fake_quantize_linear`）+ **25 个 pytest**，全过。

> **坑**：PyTorch GPU 不支持 int32 matmul，reference 的 int32 累加必须放 CPU 算。

### P2 — Triton kernel 三版迭代（单变量原则）
- **v1 naive**：固定 `BLOCK=64`，作为 baseline
- **v2.1 autotuned**：22 个 config，**kernel 函数体一字未改**，只加 autotune
- **v3 swizzled**：autotune + GROUP_M swizzle —— **Negative Result**

> **核心纪律**：v1 → v2 → v2.1 → v3 每一步只改一个变量，kernel 算法体保持 `max_diff = 0`。

### P4 — 集成进 Qwen3-0.6B
- `W8A8Linear`：`nn.Linear` 的替代品，接口完全兼容
- `convert.py`：遍历模型把所有 `nn.Linear` 换成 `W8A8Linear`（**模型无关**）
- 结果：**196 个 Linear 全部替换，1 个 lm_head 跳过，0 失败**
- **PPL: fp16=10.954, W8A8=11.303, ratio=1.032x**（优秀，接近 AWQ/GPTQ）

### P5 — 深度 Profiling
- `nsys` 看 timeline（kernel 间 gap、wrapper overhead、autotune 过程）
- `ncu` 看 hardware counter（Roof%、SM throughput、memory hierarchy）
- **侦破了一个"FillFunctor 异常"**（详见第 7 节）

---

## 4. 最终性能数据

### RTX 5090（家用，Triton 3.6）
| 形状 | naive | autotuned | cuBLAS fp16 | Roof% | vs cuBLAS |
|---|---|---|---|---|---|
| S1 (M=1) | 32.5us | 28.5us | 45.2us | 31.9% | **1.58x** ✓ |
| S2 (M=16) | 33.9us | 28.6us | 24.6us | 33.1% | 0.86x（wrapper bound）|
| S3 (M=512) | 85.1us | 62.0us | 98.5us | 32.8% | **1.57x** ✓ |
| S4 (M=2048) | 250.9us | 160.5us | 352.1us | 50.5% | **2.17x** ✓⭐ |

### 跨硬件对比：同样的 kernel，L40S 上 Roof% 更高
| 指标 | RTX 5090 (sm_120) | L40S (sm_89) |
|---|---|---|
| S4 Roof% | **50.5%** | **81%+** |
| Triton 后端 | 映射到 sm_80（Ampere） | 原生 sm_89 |

> **反直觉发现**：硬件更强的 5090 反而 Roof% 更低。原因是 Triton 把 sm_120 当 sm_80 编译，
> 没用上 Blackwell 的新指令；而 L40S 是 Triton 原生支持的目标。
> **瓶颈在 Triton 编译路径，不在硬件或我的 kernel 设计。**

> **另一个发现**：从 Triton 3.0 升到 3.6，5090 上 S4 Roof% 从 43% → 50.5%（白嫖 ~18%）。
> 软件栈版本对性能的影响经常被低估。

---

## 5. 我学到了什么（技术层面）

1. **量化的物理误差有下限**。W8A8 的 mean rel diff 在 2-6% 是正常的，不是 bug。
   per-channel 权重量化误差 ≈ `max_abs / 254 / std`，逐元素 ~1.5-2.5%。

2. **精度评估别用错指标**。`max rel diff` 会被近零分母放大成天文数字（我见过 256x）。
   真正稳健的是 **cosine similarity（>0.999）、SNR（>25dB）、PPL**。
   `tensor.quantile()` 对 >1600 万元素会报错，要用 `.median()` 替代。

3. **小 M 形状的并行度只能靠 N 维堆**。M=1/16 时 `grid_m=1`，必须用小 `BLOCK_N`（如 32）
   才能给 GPU 足够多的 program 填满 SM。我扫了 144 个 config 才发现 BN=32 系列的重要性。

4. **wrapper overhead 在小形状上是主导**。S2 上 kernel 本身 28us，但 wrapper（Python +
   Triton launch + 激活量化）能加到 50us。0.5B 模型上 W8A8 速度无优势就是这个原因，
   优势要 7B+ 才显现。

5. **Roofline 的 "Memory Throughput" ≠ DRAM Throughput**。ncu 里它是
   `max(L1%, L2%, DRAM%)`。我见过 "Memory 70% 但 DRAM 只有 3%"——意思是数据全在 cache 命中，
   根本没动 DRAM，是 compute-bound 的表现。

6. **GPU 内存层级**：DRAM → L2(96MB) → L1 → SM → Tensor Core，每层有各自带宽，
   Roofline 关心最低的那层。

7. **抽象边界的价值**：`W8A8Linear` 和 `convert.py` 完全不知道 "Qwen" 存在（不 import
   transformers）。所以从 Qwen2.5 切到 Qwen3 **只改了 test 文件一个文件**，
   kernel 和集成层一行没动。这就是生产级量化库的标准分层。

---

## 6. 我意识到了什么（方法论层面）

> 这部分是这个项目最值钱的收获，比任何性能数字都重要。

1. **Profiling 必须问题驱动**。先有一个具体疑问，再选最小工具回答它。
   不要"先 profile 看看再说"——那会拿到 100MB 看不懂的数据，浪费两小时。
   - nsys = 找"哪个 kernel 慢"
   - ncu = 找"这个 kernel 为什么慢"

2. **区分"被测对象的开销"和"profiling 工具自身的开销"**。
   这是 P5 最大的教训（见第 7 节侦探故事）。否则会优化错方向。

3. **ncu replay 的 duration 不可信**。ncu 为采集 counter 会 replay 单 kernel 17-100 次，
   把 wall-clock 放大 5-10x。**throughput% / cache hit / occupancy 可信，绝对时间不可信。**
   要用 nsys 交叉验证。

4. **NVTX 标记是隔离 Python↔GPU kernel 的最强武器**。在每个 Python 操作前后打 NVTX range，
   nsys timeline 就能让每个 GPU kernel 对齐到具体的 Python 调用，瞬间排除假说。

5. **怀疑反常结果**。当我看到"量化后准确率反而比 fp16 高 19pp"时，立刻知道是 benchmark bug
   （答案提取逻辑取错了数字），而不是"W8A8 神奇地变好了"。**量化是有损的，反常即有 bug。**

6. **用算术验证假说**。看到 142µs 的神秘 kernel，`142µs × 1792 GB/s ≈ 256 MB`，
   恰好是 Triton L2 flush buffer 的大小——一个除法就锁定了元凶。

7. **单变量改动**。每个版本只改一个东西，才能把性能变化归因到具体改动。

8. **经过严格验证的 Negative Result 比炫耀型结论更有工程价值**。
   v3 swizzle 失败，但我搞清楚了**为什么**失败（5090 96MB L2 + Triton sm_80 后端无 TMA +
   grid 太小不分批），这比"我加了 swizzle 涨了 30%"更有信息量。

---

## 7. 附录：FillFunctor 侦探案（P5 的完整 debug 故事）

**现象**：nsys 显示 `vectorized_elementwise_kernel<FillFunctor<int>>` 出现 4445 次，
avg 142µs，**占 28% GPU 时间，几乎等于 W8A8 kernel 本身**。

**排查过程**：
1. 先在 ncu 里注意到它，但意识到 **ncu replay 会放大 duration**，改用 nsys 重测拿真实数据。
2. 用 **NVTX 标记** 隔离每个 Python 操作，排除了 randn/randint/cuBLAS 等假说。
3. **算术验证**：`142µs × 1792 GB/s ≈ 254 MB`，强烈暗示是 256MB 的 "L2 cache flush buffer"。
4. **追源码**：
   - `triton/testing.py:152` → `get_empty_cache_for_benchmark()`
   - `triton/backends/nvidia/driver.py:754` → `cache_size = 256*1024*1024`,
     `torch.empty(cache_size//4, dtype=torch.int)`
   - `do_bench` 在每次 measurement 前 `cache.zero_()`（触发 `FillFunctor<int>`）

**结论**：这是 **Triton benchmark 工具（`do_bench`）为模拟冷 cache 而做的 L2 flush**，
**不是 W8A8 kernel 自身的开销**。Triton 官方注释直说："make sure that the L2 cache
doesn't contain any input data before the run"。

**影响范围**：
- ❌ 不影响 production inference（无 autotune cache miss 时不触发）
- ❌ 不影响 Qwen3 集成测试（PPL 1.032x 数据干净）
- ✅ 且 `clear_cache` 在 `start_event.record()` 之前，**不计入 kernel 计时**——
  所以我的 bench 数字一直是干净的。

---

## 8. 还能进一步做什么

按"工程价值 / 求职价值"排序：

### 短期（半天 ~ 1 天）
- [ ] **写技术博客**：把 FillFunctor 侦探案 + L40s vs 5090 Roof% 之谜整理成文章。
      这两个故事展示的是"工程师 vs 调参侠"的本质区别。
- [ ] **试 Qwen3-4B**（5090 显存够）：0.6B 太小，fp16 算术才 6%，看不出 W8A8 影响。
      4B 上 baseline 应该 40-60%，benchmark 信号会强 10 倍。

### 中期（几天）
- [ ] **dtype 泛化**：让 kernel 支持 W8A16、W4A16，模板化支持 fp32/bf16/fp16
      （课堂资料里的 checklist 第 5 条，我目前只做了 fp16 输出）。
- [ ] **消除 wrapper overhead**：把激活量化融进 kernel（quantize + matmul + dequant fusion），
      省掉中间 tensor。这是所有形状都受益的优化，比纠结 S4 Roof% 更值。
- [ ] **真实 ncu 测 Roof%**：目前 S4 的 Roof% 是 bench 推算的，可以用 ncu hardware counter
      精确测，并分析那剩下的 50% 时间 SM 在等什么（warp stall 分解）。

### 长期（方向性）
- [ ] **CUDA + pybind 对照版**：把同一个 W8A8 GEMM 用 CUDA C++ 重写，对比性能/开发时间/代码量。
      简历能同时写"熟悉 Triton 和 CUDA"。成本 1 周。
- [ ] **vLLM PR**：把这个 kernel 或学到的东西贡献到 vLLM。这是比"单算子优化"高 5x 价值的事。
- [ ] **追 Triton sm_120 native**：如果 Triton 4.x 上了 Blackwell 原生后端，5090 上 Roof%
      理论能冲到 70-80%。这个故事还没结束。

---

## 9. 仓库结构

```
QuantizedGemm/
├── reference/
│   └── torch_reference.py          # 5 个参考函数（int32 累加在 CPU）
├── triton_kernel/
│   ├── w8a8_naive.py               # v1 固定 BLOCK=64
│   ├── w8a8_autotuned.py           # v2.1 ⭐ 生产版（22 configs）
│   └── w8a8_swizzled.py            # v3 Negative Result
├── tests/
│   └── test_correctness.py         # 25 个 pytest
├── benchmarks/
│   ├── bench_kernel.py             # 主 benchmark（S1-S4 + Roof%）
│   └── torch_bench.py
├── integration/
│   ├── w8a8_linear.py              # nn.Linear 替代品
│   ├── convert.py                  # 模型 Linear 替换工具（模型无关）
│   └── diagnose_precision.py       # 多指标精度诊断（cos_sim/SNR/PPL）
├── tools/
│   ├── diagnose_autotune.py
│   ├── sweep_configs.py            # 144 config 扫描（发现 BN=32）
│   ├── profile_target.py           # 单形状 ncu 用
│   └── profile_target_nvtx.py      # NVTX 标记版
└── experiments/
    ├── qwen_w8a8_test.py           # 端到端（PPL 1.032x）
    └── qwen_benchmark_test.py      # 任务级准确率
```

---

## 10. 关键命令速查

```bash
# 跑正确性测试
uv run tests/test_correctness.py

# 跑性能 benchmark（S1-S4）
uv run benchmarks/bench_kernel.py

# Qwen3-0.6B 端到端集成测试
uv run experiments/qwen_w8a8_test.py

# ─── Profiling ───
# 1. 先停 DCGM（云环境会占 profiling counter）
sudo systemctl stop nvidia-dcgm

# 2. nsys 看 timeline（带 NVTX）
nsys profile -o trace_nvtx --force-overwrite true --trace=cuda,nvtx \
    uv run tools/profile_target_nvtx.py --shape S4

# 3. nsys kernel 排行榜（最快的诊断）
nsys stats --report cuda_gpu_kern_sum trace_nvtx.nsys-rep | head -30

# 4. ncu 看单 kernel Roof%（先 warmup 建 autotune cache）
uv run tools/profile_target.py --warmup-only --shape S4
ncu --target-processes all --kernel-name "regex:w8a8" \
    --launch-skip 0 --launch-count 2 --set roofline \
    --export profile_S4 --force-overwrite \
    uv run tools/profile_target.py --shape S4

# 5. 看 ncu 结果
ncu --import profile_S4.ncu-rep --print-summary per-kernel
```

---

## 一句话总结

> 这个项目让我从"会调 kernel 参数"变成"有完整算子工程方法论"。
> 最值钱的不是 2.17x 的加速，而是学会了：**问题驱动地 profile、用算术验证假说、
> 怀疑反常数据、区分工具开销和真实开销、以及把一个 Negative Result 讲成有信息量的故事。**