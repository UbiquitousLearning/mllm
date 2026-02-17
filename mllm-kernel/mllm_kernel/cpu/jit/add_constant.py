# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# Add constant kernel using Highway SIMD.

from __future__ import annotations

import torch

from mllm_kernel.jit_utils import jit


@jit(
    args=1,
    device="cpu",
    cpp_files=["add_constant.cpp"],
    cpp_wrappers=[("add_constant", "mllm_kernel::cpu::add_constant<1>")],
    func_name="add_constant",
)
def _add_constant_1(compiled_module, dst: torch.Tensor, src: torch.Tensor) -> None:
    compiled_module.add_constant(dst, src)


@jit(
    args=2,
    device="cpu",
    cpp_files=["add_constant.cpp"],
    cpp_wrappers=[("add_constant", "mllm_kernel::cpu::add_constant<2>")],
    func_name="add_constant",
)
def _add_constant_2(compiled_module, dst: torch.Tensor, src: torch.Tensor) -> None:
    compiled_module.add_constant(dst, src)


@jit(
    args=4,
    device="cpu",
    cpp_files=["add_constant.cpp"],
    cpp_wrappers=[("add_constant", "mllm_kernel::cpu::add_constant<4>")],
    func_name="add_constant",
)
def _add_constant_4(compiled_module, dst: torch.Tensor, src: torch.Tensor) -> None:
    compiled_module.add_constant(dst, src)


@jit(
    args=8,
    device="cpu",
    cpp_files=["add_constant.cpp"],
    cpp_wrappers=[("add_constant", "mllm_kernel::cpu::add_constant<8>")],
    func_name="add_constant",
)
def _add_constant_8(compiled_module, dst: torch.Tensor, src: torch.Tensor) -> None:
    compiled_module.add_constant(dst, src)


@jit(
    args=16,
    device="cpu",
    cpp_files=["add_constant.cpp"],
    cpp_wrappers=[("add_constant", "mllm_kernel::cpu::add_constant<16>")],
    func_name="add_constant",
)
def _add_constant_16(compiled_module, dst: torch.Tensor, src: torch.Tensor) -> None:
    compiled_module.add_constant(dst, src)


@jit(
    device="cpu",
    cpp_files=["add_constant.cpp"],
    cpp_wrappers=[("add_constant_runtime", "mllm_kernel::cpu::add_constant_runtime")],
    func_name="add_constant_runtime",
)
def _add_constant_runtime(
    compiled_module, dst: torch.Tensor, src: torch.Tensor, constant: float
) -> None:
    compiled_module.add_constant_runtime(dst, src, float(constant))


_ADD_CONSTANT_DISPATCH = {
    1: _add_constant_1,
    2: _add_constant_2,
    4: _add_constant_4,
    8: _add_constant_8,
    16: _add_constant_16,
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
    _ADD_CONSTANT_DISPATCH[constant](dst, src)
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
    _add_constant_runtime(dst, src, float(constant))
    return dst
