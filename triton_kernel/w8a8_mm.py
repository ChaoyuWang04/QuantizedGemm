"""
W8A8 scaled matmul 的 Triton kernel 入口占位文件。

当前项目还处在 P1/P2 交界,这里先记录 Triton 实现计划,并显式抛出
NotImplementedError,避免调用方误以为 kernel 已可用。

计划:
    v1: 朴素 Triton matmul,手动 int32 累加并应用 scale。
    v2: 加 @triton.autotune 搜索 BLOCK_M/BLOCK_N/BLOCK_K 等参数。
    v3: 根据 Nsight 结果优化访存、swizzle 和可能的 split-K。

参考:
    https://huggingface.co/kernels-community/triton-scaled-mm
    https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/w8a8_utils.py
"""

# 当前文件只是 P2 的实现占位;导入时直接失败可以暴露误用。
raise NotImplementedError("Triton kernel implementation pending (Stage P2)")
