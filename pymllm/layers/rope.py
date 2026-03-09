from __future__ import annotations

from typing import List, Optional, Tuple

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
            q,
            k,
            indptr,
            offsets,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        return None

    return flashinfer.rope.apply_rope(
        q,
        k,
        indptr,
        offsets,
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
            q,
            k,
            indptr,
            offsets,
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
        q,
        k,
        indptr,
        offsets,
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
            q,
            k,
            pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )
        return None

    return flashinfer.rope.apply_rope_pos_ids(
        q,
        k,
        pos_ids,
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
            q,
            k,
            pos_ids,
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
        q,
        k,
        pos_ids,
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
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox=is_neox,
        )
        return None

    return flashinfer.rope.apply_rope_with_cos_sin_cache(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox=is_neox,
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension into the first half (neox-style)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    mrope_section: List[int],
    mrope_interleaved: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply multi-dimensional rotary position embedding (M-RoPE).

    Used by Qwen3-VL which assigns independent (t, h, w) position indices to
    each token.  For text tokens all three indices are the same sequential
    value; for image tokens they follow the spatial grid layout.

    Args:
        q: Query tensor, shape ``(T, num_q_heads, head_dim)``.
        k: Key tensor, shape ``(T, num_kv_heads, head_dim)``.
        positions: 3-D position IDs, shape ``(3, T)`` — rows are
            ``(temporal, height, width)`` position indices.
        cos_sin_cache: Precomputed cache, shape ``(max_pos, head_dim)``.
            The first ``head_dim // 2`` columns are cosine values and the
            remaining columns are sine values, each for frequencies
            ``0, 1, ..., head_dim // 2 - 1``.
        mrope_section: Three integers ``[s_t, s_h, s_w]`` that partition
            the ``head_dim // 2`` rotary frequency dimensions among the
            temporal, height, and width components.
            ``sum(mrope_section)`` must equal ``head_dim // 2``.
        mrope_interleaved: When ``True`` (Qwen3-VL default), uses the
            interleaved layout where frequency dimensions are cycled
            ``(t, h, w, t, h, w, ...)`` rather than grouped consecutively.

    Returns:
        ``(q_rope, k_rope)`` with the same shapes as the inputs.
    """
    rotary_dim = cos_sin_cache.shape[-1]  # = head_dim
    half_dim = rotary_dim // 2

    # Look up cos/sin for each of the 3 position dimensions.
    # positions: [3, T]  =>  cos_sin: [3, T, rotary_dim]
    cos_sin = cos_sin_cache[positions]
    cos = cos_sin[..., :half_dim]  # [3, T, half_dim]
    sin = cos_sin[..., half_dim:]  # [3, T, half_dim]

    if mrope_interleaved:
        # Interleaved layout (Qwen3-VL): within the first
        # mrope_section[1]*3 frequency dims, indices cycle (t, h, w).
        # Remaining dims (indices >= span) all use the temporal position.
        # Matches SGLang's apply_interleaved_rope.
        cos_merged = cos[0].clone()  # start with temporal; shape [T, half_dim]
        sin_merged = sin[0].clone()
        span_h = mrope_section[1] * 3
        span_w = mrope_section[2] * 3
        cos_merged[..., 1:span_h:3] = cos[1, ..., 1:span_h:3]
        cos_merged[..., 2:span_w:3] = cos[2, ..., 2:span_w:3]
        sin_merged[..., 1:span_h:3] = sin[1, ..., 1:span_h:3]
        sin_merged[..., 2:span_w:3] = sin[2, ..., 2:span_w:3]
    else:
        # Non-interleaved (Qwen2-VL style): consecutive frequency sections.
        cos_sects = cos.split(mrope_section, dim=-1)  # list of [T, s_i]
        sin_sects = sin.split(mrope_section, dim=-1)
        # Section i picks its cos/sin from positions[i]
        cos_merged = torch.cat(
            [cos_sects[i][i] for i in range(3)], dim=-1
        )  # [T, half_dim]
        sin_merged = torch.cat(
            [sin_sects[i][i] for i in range(3)], dim=-1
        )  # [T, half_dim]

    # Expand to full rotary_dim for the neox-style rotation formula:
    # q_rot = q * cos_full + rotate_half(q) * sin_full
    cos_full = cos_merged.repeat(1, 2)  # [T, rotary_dim]
    sin_full = sin_merged.repeat(1, 2)  # [T, rotary_dim]
    cos_4d = cos_full.unsqueeze(1)  # [T, 1, rotary_dim] -- broadcasts over heads
    sin_4d = sin_full.unsqueeze(1)

    q_rot = q[..., :rotary_dim] * cos_4d + _rotate_half(q[..., :rotary_dim]) * sin_4d
    k_rot = k[..., :rotary_dim] * cos_4d + _rotate_half(k[..., :rotary_dim]) * sin_4d

    q_out = (
        torch.cat([q_rot, q[..., rotary_dim:]], dim=-1)
        if rotary_dim < q.shape[-1]
        else q_rot
    )
    k_out = (
        torch.cat([k_rot, k[..., rotary_dim:]], dim=-1)
        if rotary_dim < k.shape[-1]
        else k_rot
    )
    return q_out, k_out
