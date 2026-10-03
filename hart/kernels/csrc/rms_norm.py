import torch
import os
from torch.utils.cpp_extension import load
path = os.path.dirname(os.path.abspath(__file__)) + "/"
# Load the CUDA extension at runtime
NVCC_FLAGS = [
    "-O2",
    "-std=c++17",
    "-DENABLE_BF16",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--threads=8",
]
CXX_FLAGS = ["-O3", "-std=c++17", "-fopenmp", "-lgomp"]

rms_norm_cuda = load(
    name="rms_norm",
    sources=[path +"rms_norm.cu",path+"rms_norm.cpp"],
    extra_cflags=CXX_FLAGS,
    extra_cuda_cflags=NVCC_FLAGS,
    verbose=True,
)

def rms_norm(output: torch.Tensor,input: torch.Tensor, weight: torch.Tensor, epsilon: float, use_quant: bool = False):
    """
    Calls the CUDA RMSNorm kernel.

    Args:
        input (torch.Tensor): Input tensor [..., hidden_size].
        weight (torch.Tensor): Scale tensor [hidden_size].
        epsilon (float): Small constant for numerical stability.
        use_quant (bool): Whether to use quantization (int8 output).

    Returns:
        torch.Tensor: Normalized output tensor.
    """
    hidden_size = input.shape[-1]
    output_dtype = torch.int8 if use_quant else input.dtype
    # output = torch.empty_like(input, dtype=output_dtype)

    rms_norm_cuda.rms_norm(output, input, weight, epsilon, use_quant)
    return output
