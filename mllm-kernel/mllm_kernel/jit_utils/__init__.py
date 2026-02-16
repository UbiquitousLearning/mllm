# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# JIT compilation utilities for mllm kernels.

from .compile import (
    cache_once,
    make_cpp_args,
    load_cpu_jit,
    load_cuda_jit,
    # Path constants
    MLLM_KERNEL_TOP_PATH,
    MLLM_KERNEL_INCLUDE_DIR,
    MLLM_KERNEL_CPU_PATH,
    MLLM_KERNEL_CUDA_PATH,
    MLLM_KERNEL_ASCEND_PATH,
    MLLM_KERNEL_CPU_CSRC_DIR,
    MLLM_KERNEL_CPU_INCLUDE_DIR,
    MLLM_KERNEL_CUDA_CSRC_DIR,
    MLLM_KERNEL_CUDA_INCLUDE_DIR,
    MLLM_KERNEL_ASCEND_CSRC_DIR,
    MLLM_KERNEL_ASCEND_INCLUDE_DIR,
)

__all__ = [
    "cache_once",
    "make_cpp_args",
    "load_cpu_jit",
    "load_cuda_jit",
    "MLLM_KERNEL_TOP_PATH",
    "MLLM_KERNEL_INCLUDE_DIR",
    "MLLM_KERNEL_CPU_PATH",
    "MLLM_KERNEL_CUDA_PATH",
    "MLLM_KERNEL_ASCEND_PATH",
    "MLLM_KERNEL_CPU_CSRC_DIR",
    "MLLM_KERNEL_CPU_INCLUDE_DIR",
    "MLLM_KERNEL_CUDA_CSRC_DIR",
    "MLLM_KERNEL_CUDA_INCLUDE_DIR",
    "MLLM_KERNEL_ASCEND_CSRC_DIR",
    "MLLM_KERNEL_ASCEND_INCLUDE_DIR",
]
