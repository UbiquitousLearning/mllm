"""Chunkwise-parallel GDN extend CUDA JIT kernel (WY representation).

Implements the same GDN prefill recurrence as ``gdn_extend`` but uses
a WY-factored chunkwise decomposition inside the kernel:

  S_T = γ_C · (S_0 (I − W K^T) + Ũ K^T)

The sequential WY scan runs inside the CUDA block (shared memory), while
the per-token output and the final state update are expressed as
matrix-like operations that exploit more GPU parallelism, especially for
long sequences.

Performance notes (Jetson Orin NX, SM87):
  • Eliminates all Python-level dispatch overhead vs. the PyTorch
    chunkwise backend (``gdn_chunkwise.py``).
  • For T ≳ 4 × CHUNK_C (≈ 128 tokens) performance is comparable to
    the sequential ``gdn_extend`` kernel.
  • For T ≳ 512 tokens the smem K/W reuse begins to pay off.
  • For T < 64 the WY construction overhead dominates; prefer the
    sequential ``mllm_kernel`` backend.

Usage::

    from mllm_kernel.cuda.jit.gdn_extend_chunkwise import gdn_extend_chunkwise

    output = gdn_extend_chunkwise(
        q, k, v, a, b, A_log, dt_bias,
        state_pool, cache_indices, cu_seqlens,
    )
"""

from __future__ import annotations

import torch

from mllm_kernel.jit_utils import cache_once, jit


@cache_once
def _make_gdn_extend_chunkwise_kernel():
    """JIT-compile the chunkwise GDN extend CUDA kernel.

    On Jetson Orin (ARM CPU) the first compilation takes ~10-15 s.
    Use :func:`precompile` during server startup to avoid this latency
    on the first actual inference request.
    """

    @jit(
        args=[],
        device="cuda",
        cuda_files=["gdn_extend_chunkwise.cuh"],
        cpp_wrappers=[],
        cuda_wrappers=[
            ("gdn_extend_chunkwise", "GDNExtendChunkwiseKernel::run"),
        ],
        func_name="gdn_extend_chunkwise",
    )
    def _kernel(
        compiled_module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state_pool: torch.Tensor,
        cache_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        compiled_module.gdn_extend_chunkwise(
            q, k, v, a, b, A_log, dt_bias,
            state_pool, cache_indices, cu_seqlens, output,
        )

    return _kernel


def precompile() -> bool:
    """Eagerly trigger CUDA kernel compilation.

    Call this during server / model initialization so that the first
    inference request does not pay the compilation cost (~10-15 s on
    Jetson Orin ARM CPU).  Idempotent: subsequent calls return
    immediately because the result is cached by ``@cache_once``.

    Returns
    -------
    bool
        ``True`` if the kernel compiled successfully, ``False`` on error.
    """
    import logging
    _log = logging.getLogger(__name__)
    try:
        _make_gdn_extend_chunkwise_kernel()
        return True
    except Exception as e:
        _log.warning(
            "gdn_extend_chunkwise precompile FAILED: %s: %s",
            type(e).__name__, e, exc_info=True,
        )
        return False


def gdn_extend_chunkwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_pool: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Chunkwise-parallel GDN extend using WY decomposition (CUDA kernel).

    Processes variable-length token sequences (prefill stage).  The
    recurrent state is kept in registers across chunks, with the
    intra-chunk WY scan performed in shared memory.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor ``(total_tokens, num_k_heads, head_k_dim)``, bf16/fp16.
        L2 normalization + 1/√K scaling applied inside the kernel.
    k : torch.Tensor
        Key tensor ``(total_tokens, num_k_heads, head_k_dim)``, bf16/fp16.
    v : torch.Tensor
        Value tensor ``(total_tokens, num_v_heads, head_v_dim)``, bf16/fp16.
    a : torch.Tensor
        Decay gate input ``(total_tokens, num_v_heads)``, bf16/fp16.
    b : torch.Tensor
        Update gate input ``(total_tokens, num_v_heads)``, bf16/fp16.
    A_log : torch.Tensor
        Log-space decay parameter ``(num_v_heads,)``, float32.
    dt_bias : torch.Tensor
        Decay gate bias ``(num_v_heads,)``, float32.
    state_pool : torch.Tensor
        Pooled recurrent state
        ``(pool_size, num_v_heads, head_v_dim, head_k_dim)``, float32.
        Modified in-place.
    cache_indices : torch.Tensor
        Pool indices per request ``(batch_size,)``, int64.
    cu_seqlens : torch.Tensor
        Cumulative sequence lengths ``(batch_size + 1,)``, int64.

    Returns
    -------
    torch.Tensor
        Output ``(total_tokens, num_v_heads, head_v_dim)``, same dtype as ``v``.
    """
    total_tokens = q.shape[0]
    num_v_heads  = v.shape[1]
    head_v_dim   = v.shape[2]

    output = torch.empty(
        total_tokens, num_v_heads, head_v_dim,
        dtype=v.dtype, device=v.device,
    )

    kernel = _make_gdn_extend_chunkwise_kernel()
    kernel(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        b.contiguous(),
        A_log.contiguous(),
        dt_bias.contiguous(),
        state_pool,
        cache_indices.to(torch.int64).contiguous(),
        cu_seqlens.to(torch.int64).contiguous(),
        output,
    )
    return output
