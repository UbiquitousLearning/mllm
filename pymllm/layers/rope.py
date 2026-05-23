from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import flashinfer

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional outside CUDA builds.
    triton = None
    tl = None


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


if triton is not None:

    @triton.jit
    def _triton_mrope_forward_fused(
        q_ptr,
        k_ptr,
        cos_sin_cache_ptr,
        positions_ptr,
        q_stride,
        k_stride,
        positions_stride,
        n_qh: tl.constexpr,
        n_kh: tl.constexpr,
        hd: tl.constexpr,
        rd: tl.constexpr,
        pad_n_qh: tl.constexpr,
        pad_n_kh: tl.constexpr,
        pad_hd: tl.constexpr,
        mrope_section_t: tl.constexpr,
        mrope_section_h: tl.constexpr,
        mrope_section_w: tl.constexpr,
        is_interleaved: tl.constexpr,
        is_neox_style: tl.constexpr,
    ):
        pid = tl.program_id(0)
        q_ptr = q_ptr + pid * q_stride
        k_ptr = k_ptr + pid * k_stride
        half_rd = rd // 2

        t = tl.load(positions_ptr + pid)
        h = tl.load(positions_ptr + positions_stride + pid)
        w = tl.load(positions_ptr + 2 * positions_stride + pid)

        t_cos = cos_sin_cache_ptr + t * rd
        h_cos = cos_sin_cache_ptr + h * rd
        w_cos = cos_sin_cache_ptr + w * rd
        t_sin = t_cos + half_rd
        h_sin = h_cos + half_rd
        w_sin = w_cos + half_rd

        cos_offsets = tl.arange(0, pad_hd // 2)
        if is_interleaved:
            h_mask = ((cos_offsets % 3) == 1) & (
                cos_offsets <= 3 * mrope_section_h
            )
            w_mask = ((cos_offsets % 3) == 2) & (
                cos_offsets <= 3 * mrope_section_w
            )
            t_mask = ~(h_mask | w_mask)
        else:
            t_end = mrope_section_t
            h_end = t_end + mrope_section_h
            t_mask = cos_offsets < mrope_section_t
            h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
            w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)

        t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0.0)
        t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0.0)
        h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0.0)
        h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0.0)
        w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0.0)
        w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0.0)
        cos_row = t_cos_row + h_cos_row + w_cos_row
        sin_row = t_sin_row + h_sin_row + w_sin_row

        if is_neox_style:
            qh = tl.arange(0, pad_n_qh)[:, None]
            kh = tl.arange(0, pad_n_kh)[:, None]
            dim = tl.arange(0, pad_hd // 2)[None, :]
            q_mask = (qh < n_qh) & (dim < half_rd)
            k_mask = (kh < n_kh) & (dim < half_rd)
            q_first = qh * hd + dim
            k_first = kh * hd + dim
            q_second = q_first + half_rd
            k_second = k_first + half_rd

            q1 = tl.load(q_ptr + q_first, mask=q_mask, other=0.0).to(cos_row.dtype)
            q2 = tl.load(q_ptr + q_second, mask=q_mask, other=0.0).to(cos_row.dtype)
            k1 = tl.load(k_ptr + k_first, mask=k_mask, other=0.0).to(cos_row.dtype)
            k2 = tl.load(k_ptr + k_second, mask=k_mask, other=0.0).to(cos_row.dtype)

            tl.store(q_ptr + q_first, q1 * cos_row - q2 * sin_row, mask=q_mask)
            tl.store(q_ptr + q_second, q2 * cos_row + q1 * sin_row, mask=q_mask)
            tl.store(k_ptr + k_first, k1 * cos_row - k2 * sin_row, mask=k_mask)
            tl.store(k_ptr + k_second, k2 * cos_row + k1 * sin_row, mask=k_mask)
        else:
            qh = tl.arange(0, pad_n_qh)[:, None]
            kh = tl.arange(0, pad_n_kh)[:, None]
            pair = tl.arange(0, pad_hd // 2)[None, :]
            q_mask = (qh < n_qh) & (pair < half_rd)
            k_mask = (kh < n_kh) & (pair < half_rd)
            even = 2 * pair
            odd = even + 1

            q_even = tl.load(q_ptr + qh * hd + even, mask=q_mask, other=0.0).to(
                cos_row.dtype
            )
            q_odd = tl.load(q_ptr + qh * hd + odd, mask=q_mask, other=0.0).to(
                cos_row.dtype
            )
            k_even = tl.load(k_ptr + kh * hd + even, mask=k_mask, other=0.0).to(
                cos_row.dtype
            )
            k_odd = tl.load(k_ptr + kh * hd + odd, mask=k_mask, other=0.0).to(
                cos_row.dtype
            )

            tl.store(q_ptr + qh * hd + even, q_even * cos_row - q_odd * sin_row, mask=q_mask)
            tl.store(q_ptr + qh * hd + odd, q_odd * cos_row + q_even * sin_row, mask=q_mask)
            tl.store(k_ptr + kh * hd + even, k_even * cos_row - k_odd * sin_row, mask=k_mask)
            tl.store(k_ptr + kh * hd + odd, k_odd * cos_row + k_even * sin_row, mask=k_mask)

else:
    _triton_mrope_forward_fused = None


def _can_use_mrope_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> bool:
    return (
        triton is not None
        and q.is_cuda
        and k.is_cuda
        and positions.is_cuda
        and cos_sin_cache.is_cuda
        and q.dim() == 3
        and k.dim() == 3
        and positions.dim() == 2
        and q.is_contiguous()
        and k.is_contiguous()
        and q.shape[-1] == k.shape[-1]
        and cos_sin_cache.shape[-1] <= q.shape[-1]
    )


def apply_mrope_fused_(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    mrope_section: List[int],
    mrope_interleaved: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply M-RoPE in place with a Triton fused kernel when available.

    The fallback returns the reference PyTorch outputs, preserving functional
    correctness on CPU or non-contiguous inputs.
    """
    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(3, -1)
    if positions.stride(-1) != 1:
        positions = positions.contiguous()

    if not _can_use_mrope_fused(q, k, positions, cos_sin_cache):
        return apply_mrope(
            q,
            k,
            positions,
            cos_sin_cache,
            mrope_section,
            mrope_interleaved,
        )

    num_tokens, num_q_heads, head_size = q.shape
    num_kv_heads = k.shape[1]
    rotary_dim = cos_sin_cache.shape[-1]
    q_2d = q.reshape(num_tokens, num_q_heads * head_size)
    k_2d = k.reshape(num_tokens, num_kv_heads * head_size)

    pad_n_qh = triton.next_power_of_2(num_q_heads)
    pad_n_kh = triton.next_power_of_2(num_kv_heads)
    pad_hd = triton.next_power_of_2(head_size)

    _triton_mrope_forward_fused[(num_tokens,)](
        q_2d,
        k_2d,
        cos_sin_cache,
        positions,
        q_2d.stride(0),
        k_2d.stride(0),
        positions.stride(0),
        num_q_heads,
        num_kv_heads,
        head_size,
        rotary_dim,
        pad_n_qh,
        pad_n_kh,
        pad_hd,
        int(mrope_section[0]),
        int(mrope_section[1]),
        int(mrope_section[2]),
        bool(mrope_interleaved),
        True,
    )
    return q, k
