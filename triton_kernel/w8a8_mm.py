"""
Triton kernel for W8A8 scaled matmul.

To be implemented in P2 (Day 3-5).

Plan:
    v1: naive triton matmul with manual int32 accumulation + scale apply
    v2: add @triton.autotune over (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
    v3: Nsight-guided optimizations (vectorization, swizzling, split-K if beneficial)

Reference:
    https://huggingface.co/kernels-community/triton-scaled-mm
    https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/w8a8_utils.py
"""

# Placeholder — implement in P2
raise NotImplementedError("Triton kernel implementation pending (Stage P2)")
