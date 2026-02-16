# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# Add constant kernel using Highway SIMD.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mllm_kernel.jit_utils.compile import cache_once, load_cuda_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_add_constant_module(constant: int) -> Module:
    """
    JIT compile add_constant kernel with compile-time constant.

    Args:
        constant: The constant value to add (used as template parameter)

    Returns:
        Compiled JIT module with add_constant function
    """
    args = make_cpp_args(constant)
    return load_cuda_jit(
        "add_constant",
        *args,
        cuda_files=["add_constant.cuh"],
        cuda_wrappers=[("add_constant", f"add_constant<{args}>")],
    )


def add_constant(src: torch.Tensor, constant: int) -> torch.Tensor:
    """
    Add a compile-time constant to each element of a tensor using Highway SIMD.

    This version uses template specialization for the constant, which can enable
    additional compiler optimizations. Supported constants: 1, 2, 4, 8, 16.

    Args:
        src: Input tensor (must be float32 and contiguous)
        constant: Constant to add (must be one of: 1, 2, 4, 8, 16)

    Returns:
        Output tensor with same shape as input

    Example:
        >>> import torch
        >>> from mllm_kernel.cpu.jit import add_constant
        >>> x = torch.randn(1024)
        >>> y = add_constant(x, 16)  # y = x + 16
    """
    if constant not in (1, 2, 4, 8, 16):
        raise ValueError(
            f"Constant must be one of [1, 2, 4, 8, 16], got {constant}. "
            "Use add_constant_runtime for arbitrary constants."
        )

    if not src.is_contiguous():
        src = src.contiguous()

    dst = torch.empty_like(src)
    module = _jit_add_constant_module(constant)
    module.add_constant(dst, src)
    return dst
