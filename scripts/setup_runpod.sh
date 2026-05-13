#!/bin/bash
# setup_runpod.sh — Initialize RunPod RTX 4090 instance for W8A8 GEMM pipeline
# Usage: bash scripts/setup_runpod.sh
# Assumed base image: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

set -e
echo "=== W8A8 GEMM Pipeline Setup ==="

# 1. System info
echo "--- System info ---"
nvidia-smi
nvcc --version
python --version

# 2. Python deps
echo "--- Installing Python dependencies ---"
pip install --upgrade pip
pip install \
    "torch>=2.4.0" \
    "triton>=3.0.0" \
    pytest \
    pytest-benchmark \
    numpy \
    matplotlib \
    pandas \
    tabulate \
    transformers \
    datasets \
    "lm-eval-harness; python_version>='3.10'"

# 3. Verify CUDA + Triton
echo "--- Verifying CUDA and Triton ---"
python -c "
import torch
import triton
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
print(f'Triton: {triton.__version__}')
"

# 4. Clone CUTLASS (for later use, not built yet)
echo "--- Cloning CUTLASS ---"
if [ ! -d "third_party/cutlass" ]; then
    mkdir -p third_party
    git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git third_party/cutlass
fi

# 5. Install Nsight Compute (usually pre-installed on RunPod)
echo "--- Checking Nsight tools ---"
which ncu || echo "WARNING: ncu not found, install via apt or NVIDIA SDK"
which nsys || echo "WARNING: nsys not found"

# 6. Create result directories
mkdir -p benchmarks/results
mkdir -p benchmarks/nsight
mkdir -p profiling_reports

echo "=== Setup complete ==="
echo "Next: run 'pytest tests/test_reference.py -v' to verify PyTorch reference"
