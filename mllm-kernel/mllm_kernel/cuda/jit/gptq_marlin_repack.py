"""GPTQ/Compressed-Tensors Marlin repack CUDA JIT kernel."""

from __future__ import annotations

from typing import Optional

import torch

from mllm_kernel.jit_utils import cache_once, jit


def _normalize_perm(
    perm: Optional[torch.Tensor], size_k: int, device: torch.device
) -> torch.Tensor:
    if perm is None or perm.numel() == 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    if perm.device != device:
        raise ValueError("perm must live on the same device as b_q_weight")
    if perm.dtype != torch.int32:
        raise ValueError("perm must be int32")
    if perm.numel() != size_k:
        raise ValueError("perm length must equal size_k")
    if torch.any(perm < 0) or torch.any(perm >= size_k):
        raise ValueError("perm values must be in [0, size_k)")
    return perm.contiguous()


@cache_once
def _make_gptq_marlin_repack_kernel():
    """JIT-compile the GPTQ repack kernel."""

    @jit(
        args=[],
        device="cuda",
        cuda_files=["gemm/marlin/gptq_marlin_repack.cuh"],
        cpp_wrappers=[],
        cuda_wrappers=[("gptq_marlin_repack", "gptq_marlin_repack")],
        func_name="gptq_marlin_repack",
    )
    def _kernel(
        compiled_module,
        b_q_weight: torch.Tensor,
        perm: torch.Tensor,
        out: torch.Tensor,
        size_k: int,
        size_n: int,
        num_bits: int,
    ) -> None:
        compiled_module.gptq_marlin_repack(
            b_q_weight, perm, out, size_k, size_n, num_bits
        )

    return _kernel


def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: Optional[torch.Tensor],
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    """Repack GPTQ/Compressed-Tensors weights into Marlin layout."""

    pack_factor = 32 // num_bits
    tile_size = 16
    out = torch.empty(
        (size_k // tile_size, size_n * tile_size // pack_factor),
        dtype=b_q_weight.dtype,
        device=b_q_weight.device,
    )
    kernel = _make_gptq_marlin_repack_kernel()
    perm_t = _normalize_perm(perm, size_k, b_q_weight.device)
    kernel(b_q_weight, perm_t, out, size_k, size_n, num_bits)
    return out
