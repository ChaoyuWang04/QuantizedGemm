# w8a8-gemm-pipeline

> Learning project: 从 0 到 1 跑通一次完整的 GPU 算子开发 pipeline。  
> 载体: W8A8 (INT8 weights + INT8 activations) scaled matmul kernel。  
> 路线: PyTorch reference → Triton → CUTLASS → 整网集成。  
> Hardware: RTX 4090 (sm_89)。

## Why this project

这不是一个为了"做出比 cuBLAS 快的 kernel"的项目 — 那不现实。  
这是一个**为了把"单算子开发 pipeline"在 muscle memory 里跑通一次**的项目。

完整 pipeline 包含:需求定义 → 单测金标准 → 空骨架编译 → 精度对齐 → 性能优化迭代 → Nsight profiling → 整网集成 → 技术文章。

**真正的产出物不是 kernel,是这套工作流。**

## Repository layout

```
w8a8-gemm-pipeline/
├── docs/
│   ├── P0_requirements.md       # 项目宪法,所有决策的依据
│   ├── roofline_analysis.md     # 每个形状的 Roofline 推导
│   └── blog_draft.md            # 技术博客草稿
├── reference/
│   └── torch_reference.py       # PyTorch 金标准实现
├── triton_kernel/
│   ├── __init__.py
│   ├── w8a8_mm.py               # Triton kernel 实现 (多版本)
│   └── autotune_configs.py      # autotune 搜索空间
├── cutlass_kernel/
│   ├── csrc/
│   │   ├── w8a8_gemm.cu         # CUTLASS-based kernel
│   │   └── bindings.cpp         # PyTorch binding
│   ├── setup.py                 # 编译配置
│   └── __init__.py
├── tests/
│   ├── test_reference.py        # reference 自洽测试
│   ├── test_triton.py           # Triton kernel 单测
│   ├── test_cutlass.py          # CUTLASS kernel 单测
│   └── test_correctness.py      # 跨实现一致性
├── benchmarks/
│   ├── bench_kernel.py          # 单算子 benchmark 入口
│   ├── results/                 # JSON 结果存档
│   ├── roofline/
│   │   └── plot_roofline.py     # Roofline 图生成
│   └── nsight/                  # Nsight Compute / Systems 报告
├── scripts/
│   ├── setup_runpod.sh          # RunPod 环境初始化
│   └── run_all_tests.sh         # CI-like 本地全套测试
└── profiling_reports/           # ncu-rep 文件存档
```

## Quick start

```bash
# On RunPod RTX 4090 instance
bash scripts/setup_runpod.sh

# Run PyTorch reference tests
pytest tests/test_reference.py -v

# Run Triton kernel tests
pytest tests/test_triton.py -v

# Benchmark
python benchmarks/bench_kernel.py --shape 1,4096,4096 --kernel triton
```

## Status

- [x] **P0**: Requirements doc
- [ ] **P1**: PyTorch reference + test framework
- [ ] **P2**: Triton kernel
- [ ] **P3**: CUTLASS kernel
- [ ] **P4**: Network integration (optional)

## References

- vLLM `cutlass_scaled_mm`: vllm/csrc/quantization/cutlass_w8a8/
- HuggingFace `kernels-community/triton-scaled-mm`
- Siboehm SGEMM blog: how-to-optimize-gemm step by step
- CUTLASS 3.x examples: `cutlass/examples/55_hopper_mixed_dtype_gemm/` (closest analogue)
- SmoothQuant paper: per-token activation + per-channel weight quantization
