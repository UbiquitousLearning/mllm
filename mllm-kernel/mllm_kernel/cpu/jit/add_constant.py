# Copyright 2024 mllm Authors
# SPDX-License-Identifier: Apache-2.0
#
# Add constant kernel using Highway SIMD.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mllm_kernel.jit_utils.compile import cache_once, load_cpu_jit, make_cpp_args

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
    return load_cpu_jit(
        "add_constant",
        *args,
        cpp_files=["add_constant.cpp"],
        cpp_wrappers=[("add_constant", f"mllm_kernel::cpu::add_constant<{args}>")],
    )


@cache_once
def _jit_add_constant_runtime_module() -> Module:
    """
    JIT compile add_constant kernel with runtime constant.

    Returns:
        Compiled JIT module with add_constant_runtime function
    """
    return load_cpu_jit(
        "add_constant_runtime",
        cpp_files=["add_constant.cpp"],
        cpp_wrappers=[
            ("add_constant_runtime", "mllm_kernel::cpu::add_constant_runtime")
        ],
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

    if src.dtype != torch.float32:
        raise TypeError(f"Expected float32 tensor, got {src.dtype}")

    if not src.is_contiguous():
        src = src.contiguous()

    dst = torch.empty_like(src)
    module = _jit_add_constant_module(constant)
    module.add_constant(dst, src)
    return dst


def add_constant_runtime(src: torch.Tensor, constant: float) -> torch.Tensor:
    """
    Add a runtime constant to each element of a tensor using Highway SIMD.

    This version accepts any float constant at runtime, offering more flexibility
    than the template version.

    Args:
        src: Input tensor (must be float32 and contiguous)
        constant: Constant to add (any float value)

    Returns:
        Output tensor with same shape as input

    Example:
        >>> import torch
        >>> from mllm_kernel.cpu.jit import add_constant_runtime
        >>> x = torch.randn(1024)
        >>> y = add_constant_runtime(x, 3.14159)  # y = x + 3.14159
    """
    if src.dtype != torch.float32:
        raise TypeError(f"Expected float32 tensor, got {src.dtype}")

    if not src.is_contiguous():
        src = src.contiguous()

    dst = torch.empty_like(src)
    module = _jit_add_constant_runtime_module()
    module.add_constant_runtime(dst, src, float(constant))
    return dst
