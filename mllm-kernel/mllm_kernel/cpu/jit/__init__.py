# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# CPU JIT kernels with Highway SIMD support.

from .add_constant import add_constant, add_constant_runtime

__all__ = [
    "add_constant",
    "add_constant_runtime",
]
