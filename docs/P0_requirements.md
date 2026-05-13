# P0 需求文档:W8A8 GEMM Pipeline

> 本文档是接下来 7-10 天的"宪法"。任何实现决策与此冲突时,先停下来更新本文档,再继续编码。
> Last updated: 2026-05-13

---

## 0. 项目第一性目标

**真目标**:跑通一次完整的「单算子开发→优化→集成」pipeline,获得可复用的工作流。

**假目标**(不要被这些劫持):
- ❌ "超过 cuBLAS" — 不现实,也不是新人 pipeline 的目的
- ❌ "完整对齐 vLLM 主线 API" — nice to have,但不阻塞 pipeline 推进
- ❌ "覆盖所有量化粒度" — 只做 per-token + per-channel 一种

**完成定义 (Definition of Done)**:
- [ ] Triton kernel 在 LLaMA-7B 4 个形状上单测通过,精度 max abs diff < 1e-2
- [ ] CUTLASS kernel 在同样形状上单测通过
- [ ] Nsight 报告产出,Roofline 图标出每个版本位置
- [ ] 技术博客 1 篇,讲清楚"为什么 W8A8 在 decode 上没有 4× 加速"
- [ ] (可选)Qwen2.5-0.5B 整网集成验证

---

## 1. 算子规格 (Spec)

### 1.1 数学定义

给定:
- 量化激活 $A^q \in \mathbb{Z}_{[-128, 127]}^{M \times K}$ (int8)
- 量化权重 $B^q \in \mathbb{Z}_{[-128, 127]}^{K \times N}$ (int8)
- Per-token scale $s_A \in \mathbb{R}_+^{M}$ (fp32)
- Per-channel scale $s_B \in \mathbb{R}_+^{N}$ (fp32)
- 可选 bias $b \in \mathbb{R}^{N}$ (fp16)

输出:
$$
C_{ij} = \left(\sum_{k=0}^{K-1} A^q_{ik} \cdot B^q_{kj}\right) \cdot s_A^{(i)} \cdot s_B^{(j)} + b_j
$$

中间结果 (mma 累加) 在 int32 域,最后 dequant 到 fp16/bf16。

### 1.2 接口签名

**Python API**:
```python
def w8a8_scaled_mm(
    a: torch.Tensor,              # [M, K] int8, row-major
    b: torch.Tensor,              # [K, N] int8, col-major (or [N, K] row-major)
    scale_a: torch.Tensor,        # [M, 1] fp32
    scale_b: torch.Tensor,        # [1, N] fp32
    bias: Optional[torch.Tensor] = None,  # [N] fp16
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:                # [M, N] out_dtype
    ...
```

**B 矩阵布局决策**: `[K, N]` row-major (= `[N, K]` col-major) — 这是 NVIDIA Tensor Core `mma.sync m16n8k32` 对 B 操作数的天然布局。否则需要在 kernel 内做 transpose,徒增复杂度。

### 1.3 不支持范围 (out of scope)

- ❌ M=0 / K=0 / N=0 边界
- ❌ Per-tensor scale (太简单,没学习价值)
- ❌ Block-wise scale (DeepSeek FP8 路线,留给 v2)
- ❌ Asymmetric quantization (zero_point ≠ 0)
- ❌ 非 16 对齐的 N (CUTLASS 路径强制 N % 16 == 0)

---

## 2. 性能目标 (Performance Budget)

### 2.1 测试形状矩阵

| Shape | (M, N, K) | LLaMA-7B 对应场景 | 理论 bound |
|---|---|---|---|
| **S1** | (1, 4096, 4096) | decode, batch=1, q_proj | 极端 memory-bound |
| **S2** | (16, 4096, 4096) | decode, batch=16 | memory-bound |
| **S3** | (512, 4096, 4096) | prefill, seq_len=512 | balanced |
| **S4** | (2048, 4096, 4096) | prefill, seq_len=2048 | compute-bound |

可选扩展:LLaMA-7B 的 FFN 层 (N=11008, K=4096) 留到性能优化后期。

### 2.2 Roofline 预估表 (RTX 4090)

**硬件常数**:
- INT8 Tensor Core 算力 (dense): **330 TOPS**
- HBM 带宽: **1008 GB/s**
- AI 拐点: 330e12 / 1008e9 ≈ **327 ops/byte**

| Shape | AI (ops/byte) | Bound | 理论上限 | 目标 (50%) | Stretch (70%) |
|---|---|---|---|---|---|
| S1 (1×4096×4096) | ≈ 2.0 | memory | **2.0 TOPS** | 1.0 TOPS | 1.4 TOPS |
| S2 (16×4096×4096) | ≈ 30 | memory | **30 TOPS** | 15 TOPS | 21 TOPS |
| S3 (512×4096×4096) | ≈ 256 | balanced | **258 TOPS** | 129 TOPS | 181 TOPS |
| S4 (2048×4096×4096) | ≈ 327 | crossover | **330 TOPS** | 165 TOPS | 231 TOPS |

**计算说明** (S1 为例):
- 数据量: $M \cdot K + K \cdot N + M \cdot N$ bytes (int8 各 1B, output fp16 2B,简化忽略 scale)
  = $1 \cdot 4096 + 4096 \cdot 4096 + 1 \cdot 4096 \cdot 2$ ≈ 16.8 MB
- 计算量: $2MNK$ = $2 \cdot 1 \cdot 4096 \cdot 4096$ ≈ 33.6 MOps
- AI = 33.6e6 / 16.8e6 ≈ 2.0 ops/byte
- 受限于带宽: $1008 \text{ GB/s} \times 2 \text{ ops/byte}$ = 2.0 TOPS

**关键认知**: S1 怎么优化都不可能超过 2 TOPS,这不是 kernel 能力问题,是物理定律。

### 2.3 与基线对比目标

| 对照基线 | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| vs PyTorch fp16 matmul | ≥ 1.0× | ≥ 1.0× | ≥ 1.5× | ≥ 2.0× |
| vs cuBLAS fp16 (torch.mm) | ≥ 0.5× | ≥ 0.5× | ≥ 0.7× | ≥ 0.7× |
| vs vLLM `cutlass_scaled_mm` | ≥ 0.8× | ≥ 0.8× | ≥ 0.8× | ≥ 0.8× |

**注**: S1/S2 可能在 PyTorch fp16 之下,因为 PyTorch 直接用 cuBLAS;反量化开销吃掉了 INT8 算力优势。**这正是我们要在博客里揭示的现象**,不是失败。

---

## 3. 精度目标 (Numerical Budget)

### 3.1 误差容忍

参考 vLLM 单测惯例 (`vllm/tests/kernels/`):
- **max abs diff** vs PyTorch fp32 reference: ≤ 1e-2
- **mean abs diff**: ≤ 1e-3
- **相对误差** (max abs diff / max abs ref): ≤ 1%

### 3.2 输入数据分布

测试用随机数生成:
```python
torch.manual_seed(42)
# Activation: int8 in [-128, 127], uniform
a_int8 = torch.randint(-128, 128, (M, K), dtype=torch.int8)
# Weight: int8 in [-128, 127], uniform  
b_int8 = torch.randint(-128, 128, (K, N), dtype=torch.int8)
# Scales: 模拟真实分布,激活 scale 较大,权重 scale 较小
scale_a = torch.rand(M, 1, dtype=torch.float32) * 0.1  # [0, 0.1]
scale_b = torch.rand(1, N, dtype=torch.float32) * 0.01 # [0, 0.01]
```

### 3.3 边界 case 单测

| Case | 描述 | 目的 |
|---|---|---|
| `test_zero_input` | a 全零 | 验证零值不溢出 |
| `test_max_input` | a 全 127, b 全 127 | 验证 int32 累加不溢出 |
| `test_negative_input` | a 全 -128, b 全 -128 | 验证补码处理正确 |
| `test_small_shape` | M=N=K=16 | 最小可运行形状 |
| `test_non_aligned` | M=15, N=33, K=129 | 非对齐 padding 处理 |

---

## 4. Pipeline 阶段定义

### Stage P1: PyTorch 金标准 (Day 1-2)
- [ ] `reference/torch_reference.py`: 写 PyTorch 实现,先 fp32 算 int32 累加,再乘 scale
- [ ] `tests/test_correctness.py`: pytest 框架搭好,5 个边界 case + 4 个 LLaMA 形状
- [ ] 跑通 `pytest tests/test_correctness.py::test_reference` 全绿

**Definition of Done**: 跑 pytest 能输出 9 个 PASSED,reference 自己和自己对比 max_diff = 0。

### Stage P2: Triton Kernel (Day 3-5)
- [ ] `triton_kernel/w8a8_mm.py`: 写 Triton naive 版本 (v1)
- [ ] 加 `@triton.autotune` 调参 (v2)
- [ ] Nsight 看带宽利用率,迭代到 v3
- [ ] 单测全绿,性能达标

**Definition of Done**: 4 个形状性能达到表 2.3 的 50% 目标。

### Stage P3: CUTLASS Kernel (Day 6-9)
- [ ] `cutlass_kernel/csrc/w8a8_gemm.cu`: 基于 CUTLASS 3.x 写 scaled_mm
- [ ] `setup.py` 编译流程,能 `pip install -e .`
- [ ] PyTorch binding (`torch.utils.cpp_extension` 或 `pybind11`)
- [ ] 单测对齐 Triton 版本结果

**Definition of Done**: CUTLASS 版本性能 ≥ Triton 版本。

### Stage P4: 整网集成 (Day 10, 可选)
- [ ] 在 Qwen2.5-0.5B 上替换 q_proj/k_proj 的 linear
- [ ] 跑 wikitext perplexity 对比
- [ ] 端到端 latency 对比

**Definition of Done**: perplexity 差距 < 1%,latency 有可测量收益。

---

## 5. 工程约束

### 5.1 开发环境
- **硬件**: RunPod RTX 4090 单卡 (24GB)
- **CUDA**: 12.4+
- **PyTorch**: 2.4+ (匹配 CUDA 12.4)
- **Triton**: 3.0+
- **CUTLASS**: 3.5+ (Ada arch sm_89)

### 5.2 代码规范
- Python: `black` + `isort`,行宽 100
- CUDA: Google C++ style,2 空格缩进
- Commit message: conventional commits (`feat:`, `fix:`, `perf:`, `docs:`)
- 每个性能优化版本独立 commit,benchmark 数字写在 commit message 里

### 5.3 性能记录格式
所有 benchmark 结果存 `benchmarks/results/`,JSON 格式:
```json
{
  "version": "triton-v2",
  "commit": "abc1234",
  "gpu": "RTX 4090",
  "shape": [1, 4096, 4096],
  "tflops": 1.23,
  "bandwidth_gbps": 856.4,
  "bandwidth_util": 0.85,
  "latency_us": 27.3
}
```

---

## 6. 风险登记 (Risk Register)

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| CUTLASS 编译环境难搞 | 高 | 高 | 用 NVIDIA Docker 镜像 `pytorch:24.10-py3` |
| Triton autotune 时间过长 | 中 | 中 | 限制 search space,缓存结果 |
| 4090 上 INT8 实际加速不如预期 | 高 | 低 | 这就是博客要讲的发现,不是失败 |
| RunPod 4090 实例抢不到 | 中 | 中 | 备选 A6000 / L4 (都是 Ada arch) |
| 时间超 10 天 | 中 | 中 | P4 整网集成可砍,核心是 P1-P3 |

---

## 7. 不做什么 (Anti-goals)

- ❌ 不写一行没有单测的代码 — 每个 commit 都得是绿的
- ❌ 不做性能优化直到精度过 — 「先正确再快」
- ❌ 不追求"漂亮代码",追求"能讲清楚的代码" — 代码是博客的素材
- ❌ 不依赖 vLLM 仓库 — 独立仓库,自包含
- ❌ 不做 W8A16 / W4A16 / FP8 — 这次只做 W8A8

---

## 8. 元规则自检

按 user 的元规则,本文档应该:
- [x] 每一步行动清晰可执行 (规则 0)
- [x] 关键概念有数学公式 (规则 1)
- [x] Roofline 分析有"对偶概念" (memory-bound vs compute-bound)
- [x] 标出可能的"不知道自己不知道" (S1 上 INT8 没加速的反直觉真相)

如发现规则失效或文档过时,**先更新本文档,再写代码**。
