"""AWQ Marlin weight repack CUDA JIT kernel.

Repacks AWQ-format quantized weights into Marlin kernel layout.

Usage::

    from mllm_kernel.cuda.jit.awq_marlin_repack import awq_marlin_repack

    out = awq_marlin_repack(b_q_weight, size_k, size_n, num_bits)
"""

from __future__ import annotations

import torch

from mllm_kernel.jit_utils import cache_once, jit


@cache_once
def _make_awq_marlin_repack_kernel():
    """JIT-compile the AWQ Marlin repack CUDA kernel."""

    @jit(
        args=[],
        device="cuda",
        cuda_files=["gemm/marlin/awq_marlin_repack.cuh"],
        cuda_wrappers=[("awq_marlin_repack", "awq_marlin_repack")],
        func_name="awq_marlin_repack",
    )
    def _kernel(
        compiled_module,
        out: torch.Tensor,
        b_q_weight: torch.Tensor,
        size_k: int,
        size_n: int,
        num_bits: int,
    ) -> None:
        compiled_module.awq_marlin_repack(out, b_q_weight, size_k, size_n, num_bits)

    return _kernel


def awq_marlin_repack(
    b_q_weight: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    """Repack AWQ-format quantized weights into Marlin kernel layout.

    Parameters
    ----------
    b_q_weight : torch.Tensor
        AWQ packed weight tensor, shape ``(size_k, size_n // pack_factor)``,
        dtype ``int32``.
    size_k : int
        Number of input features (must be divisible by 16).
    size_n : int
        Number of output features (must be divisible by 64).
    num_bits : int
        Weight quantization bit-width (4 or 8).

    Returns
    -------
    torch.Tensor
        Repacked weight tensor in Marlin layout, shape
        ``(size_k // 16, size_n * 16 // pack_factor)``, dtype ``int32``.
    """
    tile_size = 16
    pack_factor = 32 // num_bits
    out = torch.empty(
        (size_k // tile_size, size_n * tile_size // pack_factor),
        dtype=b_q_weight.dtype,
        device=b_q_weight.device,
    )
    kernel = _make_awq_marlin_repack_kernel()
    kernel(out, b_q_weight, size_k, size_n, num_bits)
    return out
