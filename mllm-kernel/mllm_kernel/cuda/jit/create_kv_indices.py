"""High-performance CUDA JIT wrapper for create_kv_indices.

This module exposes a single function:

    create_kv_indices(req_to_token, req_pool_indices,
                      page_kernel_lens, kv_indptr,
                      kv_start_idx, kv_indices)

which is a Python binding around the C++/CUDA kernel defined in
`mllm_kernel/cuda/csrc/create_kv_indices.cuh`.

The kernel transforms pymllm's 2-D ReqToTokenPool mapping table into the flat
`(kv_indptr, kv_indices)` layout expected by FlashInfer's paged KV attention
wrappers.  It is carefully written for maximum throughput and is intended to
replace the Triton implementation `_create_kv_indices_triton` in
`pymllm.layers.attention.flashinfer_backend`.
"""

from __future__ import annotations

import torch

from mllm_kernel.jit_utils import cache_once, jit


@cache_once
def _make_create_kv_indices_kernel():
    """JIT-compile the CUDA kernel and return a callable wrapper.

    The JIT system will:
      * locate `create_kv_indices.cuh` under the mllm-kernel CUDA csrc tree,
      * compile it into a TVM FFI module,
      * expose `CreateKvIndicesKernel::run` as `compiled_module.create_kv_indices`.
    """

    @jit(
        args=[],
        device="cuda",
        cuda_files=["create_kv_indices.cuh"],
        cpp_wrappers=[],
        cuda_wrappers=[
            ("create_kv_indices", "CreateKvIndicesKernel::run"),
        ],
        func_name="create_kv_indices",
    )
    def _kernel(
        compiled_module,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        page_kernel_lens: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_start_idx: torch.Tensor,
        kv_indices: torch.Tensor,
    ) -> None:
        compiled_module.create_kv_indices(
            req_to_token,
            req_pool_indices,
            page_kernel_lens,
            kv_indptr,
            kv_start_idx,
            kv_indices,
        )

    return _kernel


def create_kv_indices(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_kernel_lens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_start_idx: torch.Tensor | None,
    kv_indices: torch.Tensor,
) -> None:
    """Fill a flat KV-index buffer from the ReqToTokenPool mapping.

    This is a thin Python wrapper that forwards to the JIT-compiled CUDA
    kernel.  All tensors must be placed on the same CUDA device.

    Args
    ----
    req_to_token:
        Mapping tensor from ReqToTokenPool, shape
        ``[max_reqs, max_context_len]``, dtype ``torch.int32``.
    req_pool_indices:
        Request slots participating in this batch, shape ``[batch_size]``,
        dtype ``torch.int32``.
    page_kernel_lens:
        Per-sequence token counts (how many tokens to attend), shape
        ``[batch_size]``, dtype ``torch.int32``.
    kv_indptr:
        Prefix sums over per-sequence token counts, shape ``[batch_size + 1]``,
        dtype ``torch.int32``.  ``kv_indptr[i]`` is the starting offset in
        ``kv_indices`` for sequence ``i``.
    kv_start_idx:
        Optional starting positions inside each sequence, shape
        ``[batch_size]`` or ``[0]``, dtype ``torch.int32``.  When
        ``None``, the kernel assumes 0 for all sequences.
    kv_indices:
        Output flat KV-index buffer, shape ``[N]``, dtype ``torch.int32``.
        ``N`` must be at least ``kv_indptr[batch_size]``.
    """
    if kv_start_idx is None:
        # Use an empty tensor to signal "no start offsets".  The C++ launcher
        # treats length==0 as "no kv_start" and will pass a nullptr into the
        # parameter block, which is slightly cheaper than materialising a
        # full zero tensor on every call.
        kv_start_idx = req_pool_indices.new_empty(0, dtype=torch.int32)

    kernel = _make_create_kv_indices_kernel()
    kernel(
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        kv_indptr,
        kv_start_idx,
        kv_indices,
    )
