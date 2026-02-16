# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# mllm-kernel: High-performance kernels for mllm
#
# This package provides JIT-compiled kernels for CPU (with Highway SIMD),
# CUDA, and Ascend platforms.

__version__ = "1.0.0"

from . import cpu
from . import jit_utils

__all__ = [
    "__version__",
    "cpu",
    "jit_utils",
]
