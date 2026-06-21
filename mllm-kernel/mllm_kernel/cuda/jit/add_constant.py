# Copyright (c) MLLM Team.
# Licensed under the MIT License.


from __future__ import annotations

import torch

from mllm_kernel.jit_utils import jit


_SUPPORTED_CONSTANTS = (1, 2, 4, 8, 16)


def _make_add_constant_kernel(constant: int):
    @jit(
        args=constant,
        cuda_files=["add_constant.cuh"],
        cpp_wrappers=[],
        cuda_wrappers=[("add_constant", f"add_constant<{constant}>")],
        device="cuda",
        func_name="add_constant",
    )
    def _kernel(compiled_module, dst: torch.Tensor, src: torch.Tensor) -> None:
        compiled_module.add_constant(dst, src)

    return _kernel


_ADD_CONSTANT_DISPATCH = {
    constant: _make_add_constant_kernel(constant) for constant in _SUPPORTED_CONSTANTS
}


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
    if constant not in _SUPPORTED_CONSTANTS:
        raise ValueError(
            f"Constant must be one of {list(_SUPPORTED_CONSTANTS)}, got {constant}. "
            "Use add_constant_runtime for arbitrary constants."
        )

    if not src.is_contiguous():
        src = src.contiguous()

    dst = torch.empty_like(src)
    _ADD_CONSTANT_DISPATCH[constant](dst, src)
    return dst
