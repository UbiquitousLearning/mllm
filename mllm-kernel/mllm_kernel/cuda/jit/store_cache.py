# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# Python interface for the store_cache CUDA kernel.
# Efficiently scatters key/value tensors into a pre-allocated KV cache pool.

from __future__ import annotations

import logging
import torch
from mllm_kernel.jit_utils import jit
from mllm_kernel.jit_utils.compile import cache_once, make_cpp_args


logger = logging.getLogger(__name__)


@cache_once
def _is_arch_support_pdl() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    # PDL requires sm_90a (Hopper) or later
    return major > 9 or (major == 9 and minor >= 0)


def _make_store_cache_kernel(row_bytes: int):
    """Create a JIT-compiled store_cache kernel for the given row_bytes."""
    pdl = _is_arch_support_pdl()
    cpp_args = make_cpp_args(row_bytes, pdl)

    @jit(
        args=[row_bytes, pdl],
        device="cuda",
        cuda_files=["store_cache.cuh"],
        cpp_wrappers=[],
        cuda_wrappers=[
            ("store_cache", f"StoreKVCacheKernel<{cpp_args}>::run"),
        ],
        func_name="store_cache",
    )
    def _kernel(
        compiled_module,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        indices: torch.Tensor,
        num_split: int,
    ) -> None:
        compiled_module.store_cache(k, v, k_cache, v_cache, indices, num_split)

    return _kernel


_KERNEL_CACHE: dict[int, object] = {}


def _get_kernel(row_bytes: int):
    if row_bytes not in _KERNEL_CACHE:
        _KERNEL_CACHE[row_bytes] = _make_store_cache_kernel(row_bytes)
    return _KERNEL_CACHE[row_bytes]


@cache_once
def can_use_store_cache(row_bytes: int) -> bool:
    """Check whether the JIT store_cache kernel supports the given row size.

    Returns ``False`` if *row_bytes* is not a multiple of 4 or if the JIT
    compilation fails for any reason.
    """
    if row_bytes % 4 != 0:
        logger.warning(
            "Unsupported row_bytes=%d for JIT store_cache kernel: "
            "must be multiple of 4",
            row_bytes,
        )
        return False
    try:
        _get_kernel(row_bytes)
        return True
    except Exception as e:
        logger.warning(
            "Failed to load JIT store_cache kernel with row_bytes=%d: %s",
            row_bytes,
            e,
        )
        return False


def store_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    row_bytes: int = 0,
    num_split: int = 0,
) -> None:
    """Store key and value tensors into a KV cache at specified indices.

    Each row of *k* (and *v*) is scattered into *k_cache* (and *v_cache*)
    at the location given by the corresponding entry in *indices*.

    Args:
        k: Key tensor, shape ``(batch_size, head_num * head_dim)``.
        v: Value tensor, shape ``(batch_size, head_num * head_dim)``.
        k_cache: Key cache, shape ``(num_slots, head_num * head_dim)``.
        v_cache: Value cache, shape ``(num_slots, head_num * head_dim)``.
        indices: Index tensor, shape ``(batch_size,)``, dtype int32 or int64.
        row_bytes: Bytes per row. Auto-detected from *k* when 0.
        num_split: Number of warps that cooperate on each element (1, 2, or 4).
            When 0 the best value is chosen automatically based on alignment.
    """
    row_bytes = row_bytes or k.shape[-1] * k.element_size()
    kernel = _get_kernel(row_bytes)

    if num_split <= 0:
        if row_bytes % 2048 == 0:
            num_split = 4
        elif row_bytes % 1024 == 0:
            num_split = 2
        else:
            num_split = 1

    kernel(k, v, k_cache, v_cache, indices, num_split)
