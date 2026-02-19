from __future__ import annotations

from typing import Optional, Tuple

import torch
import flashinfer


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    inplace: bool = False,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1.0,
    rope_theta: float = 1e4,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Apply rotary embedding to a batch of queries/keys (stored as RaggedTensor).

    cos/sin values are computed on the fly inside the kernel. Position offsets
    are provided per-segment via ``indptr`` and ``offsets``.

    Args:
        q: Query ragged tensor, shape ``(nnz, num_q_heads, head_dim)``.
        k: Key ragged tensor, shape ``(nnz, num_k_heads, head_dim)``.
        indptr: Indptr tensor, shape ``(batch_size + 1,)``. The i-th segment
            spans ``q[indptr[i]:indptr[i+1]]``.
        offsets: Relative position offsets per segment, shape ``(batch_size,)``.
        inplace: If ``True``, apply RoPE in-place and return ``None``.
            If ``False``, return new ``(q_rope, k_rope)`` tensors.
        rotary_dim: Number of dimensions to apply RoPE to.  ``None`` means
            the entire ``head_dim``.
        interleave: If ``True``, rotate even/odd dims (``[..., ::2]`` /
            ``[..., 1::2]``). If ``False``, rotate first/second half dims.
        rope_scale: Scaling factor for position indices.
        rope_theta: Base frequency theta.

    Returns:
        ``None`` when *inplace* is ``True``, otherwise a tuple
        ``(q_rope, k_rope)`` of rotated tensors with the same shapes as
        the inputs.
    """
    if inplace:
        flashinfer.rope.apply_rope_inplace(
            q, k, indptr, offsets,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        return None

    return flashinfer.rope.apply_rope(
        q, k, indptr, offsets,
        rotary_dim=rotary_dim,
        interleave=interleave,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    )


def apply_llama31_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    inplace: bool = False,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8.0,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    old_context_len: int = 8192,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Apply Llama 3.1 style rotary embedding to a batch of queries/keys.

    This variant adjusts frequencies with ``low_freq_factor``,
    ``high_freq_factor``, and ``old_context_len`` following the Llama 3.1
    RoPE recipe. cos/sin values are computed on the fly.

    Args:
        q: Query ragged tensor, shape ``(nnz, num_q_heads, head_dim)``.
        k: Key ragged tensor, shape ``(nnz, num_k_heads, head_dim)``.
        indptr: Indptr tensor, shape ``(batch_size + 1,)``.
        offsets: Relative position offsets per segment, shape ``(batch_size,)``.
        inplace: If ``True``, apply in-place and return ``None``.
        rotary_dim: Number of dimensions to apply RoPE to. ``None`` means
            the entire ``head_dim``.
        interleave: If ``True``, rotate even/odd dims; otherwise first/second
            half dims.
        rope_scale: Scaling factor for position indices (default ``8``).
        rope_theta: Base frequency theta (default ``5e5``).
        low_freq_factor: Low frequency factor for Llama 3.1 RoPE.
        high_freq_factor: High frequency factor for Llama 3.1 RoPE.
        old_context_len: Original context length for Llama 3.1 RoPE.

    Returns:
        ``None`` when *inplace* is ``True``, otherwise ``(q_rope, k_rope)``.
    """
    if inplace:
        flashinfer.rope.apply_llama31_rope_inplace(
            q, k, indptr, offsets,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
            old_context_len=old_context_len,
        )
        return None

    return flashinfer.rope.apply_llama31_rope(
        q, k, indptr, offsets,
        rotary_dim=rotary_dim,
        interleave=interleave,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        low_freq_factor=low_freq_factor,
        high_freq_factor=high_freq_factor,
        old_context_len=old_context_len,
    )


def apply_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    inplace: bool = False,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 1.0,
    rope_theta: float = 1e4,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Apply rotary embedding using explicit per-token position IDs.

    Unlike :func:`apply_rope` which derives positions from ``indptr`` /
    ``offsets``, this function takes a flat ``pos_ids`` tensor that supplies
    an explicit position for every token.

    Args:
        q: Query tensor, shape ``(nnz, num_q_heads, head_dim)``.
        k: Key tensor, shape ``(nnz, num_k_heads, head_dim)``.
        pos_ids: Position indices, shape ``(nnz,)``.
        inplace: If ``True``, apply in-place and return ``None``.
        rotary_dim: Number of dimensions to apply RoPE to.
        interleave: Interleaved layout flag.
        rope_scale: Scaling factor for position indices.
        rope_theta: Base frequency theta.

    Returns:
        ``None`` when *inplace* is ``True``, otherwise ``(q_rope, k_rope)``.
    """
    if inplace:
        flashinfer.rope.apply_rope_pos_ids_inplace(
            q, k, pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        return None

    return flashinfer.rope.apply_rope_pos_ids(
        q, k, pos_ids,
        rotary_dim=rotary_dim,
        interleave=interleave,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    )


def apply_llama31_rope_pos_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    inplace: bool = False,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8.0,
    rope_theta: float = 5e5,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    old_context_len: int = 8192,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Apply Llama 3.1 style RoPE using explicit per-token position IDs.

    Combines Llama 3.1 frequency adjustments with explicit ``pos_ids``.

    Args:
        q: Query tensor, shape ``(nnz, num_q_heads, head_dim)``.
        k: Key tensor, shape ``(nnz, num_k_heads, head_dim)``.
        pos_ids: Position indices, shape ``(nnz,)``.
        inplace: If ``True``, apply in-place and return ``None``.
        rotary_dim: Number of dimensions to apply RoPE to.
        interleave: Interleaved layout flag.
        rope_scale: Scaling factor (default ``8``).
        rope_theta: Base frequency theta (default ``5e5``).
        low_freq_factor: Low frequency factor for Llama 3.1 RoPE.
        high_freq_factor: High frequency factor for Llama 3.1 RoPE.
        old_context_len: Original context length for Llama 3.1 RoPE.

    Returns:
        ``None`` when *inplace* is ``True``, otherwise ``(q_rope, k_rope)``.
    """
    if inplace:
        flashinfer.rope.apply_llama31_rope_pos_ids_inplace(
            q, k, pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
            old_context_len=old_context_len,
        )
        return None

    return flashinfer.rope.apply_llama31_rope_pos_ids(
        q, k, pos_ids,
        rotary_dim=rotary_dim,
        interleave=interleave,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        low_freq_factor=low_freq_factor,
        high_freq_factor=high_freq_factor,
        old_context_len=old_context_len,
    )


def apply_rope_with_cos_sin_cache(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    inplace: bool = False,
    is_neox: bool = True,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Apply rotary embedding with precomputed cos/sin cache.

    Compatible with SGL/vLLM implementations. Note that ``query`` and ``key``
    use a **flattened** head layout ``(nnz, num_heads * head_size)`` instead
    of the 3-D layout used by the other ``apply_rope*`` functions.

    Args:
        positions: Position indices, shape ``(nnz,)``.
        query: Query tensor, shape ``(nnz, num_q_heads * head_size)``.
        key: Key tensor, shape ``(nnz, num_k_heads * head_size)``.
        head_size: Size of each attention head.
        cos_sin_cache: Precomputed cos/sin tensor, shape
            ``(max_seq_len, rotary_dim)``. The first half of ``rotary_dim``
            stores cosine values, the second half stores sine values.
        inplace: If ``True``, apply in-place and return ``None``.
        is_neox: If ``True`` (default), use GPT-NeoX style (rotate
            first/second half dims). If ``False``, use interleaved style
            (rotate even/odd dims).

    Returns:
        ``None`` when *inplace* is ``True``, otherwise
        ``(query_out, key_out)`` with the same shapes as the inputs.
    """
    if inplace:
        flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(
            positions, query, key, head_size, cos_sin_cache,
            is_neox=is_neox,
        )
        return None

    return flashinfer.rope.apply_rope_with_cos_sin_cache(
        positions, query, key, head_size, cos_sin_cache,
        is_neox=is_neox,
    )
