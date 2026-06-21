"""RadixAttention -- the attention layer used by pymllm models.

This module is kept small intentionally: all heavy computation is delegated
to the pluggable ``AttentionBackend`` that is attached to the ``ForwardBatch``.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

if TYPE_CHECKING:
    from pymllm.engine.forward_batch import ForwardBatch


# ---------------------------------------------------------------------------
# AttentionType
# ---------------------------------------------------------------------------


class AttentionType(Enum):
    """Attention variant used by a :class:`RadixAttention` layer.

    Uses string values so that ``torch.compile`` can treat them as constants.
    """

    # Standard causal self-attention in a decoder layer.
    DECODER = "decoder"

    # Bidirectional self-attention for image tokens inside a decoder
    # (e.g. VLM visual encoder embedded in the language model).
    DECODER_BIDIRECTIONAL = "decoder_bidirectional"

    # Full bidirectional self-attention in an encoder-only model.
    ENCODER_ONLY = "encoder_only"


# ---------------------------------------------------------------------------
# RadixAttention
# ---------------------------------------------------------------------------


class RadixAttention(nn.Module):
    """Attention layer that delegates computation to a pluggable backend.

    Each transformer attention layer in a pymllm model creates exactly one
    ``RadixAttention`` with a unique ``layer_id``.  During the forward pass
    the layer looks up the correct KV buffer via ``layer_id`` and calls the
    backend attached to the current :class:`~pymllm.engine.forward_batch.ForwardBatch`.

    Parameters
    ----------
    num_heads
        Number of query attention heads (after any tensor-parallelism
        sharding; pass the full count if not using TP).
    head_dim
        Per-head dimension for query and key projections.
    scaling
        Softmax pre-scale, typically ``1 / sqrt(head_dim)``.
    num_kv_heads
        Number of key / value heads (supports GQA / MQA).
    layer_id
        Zero-based index of this layer within the model.  Used to index into
        ``KVPool.k_buffer`` / ``v_buffer``.
    logit_cap
        If > 0, attention logits are soft-capped to this value via a ``tanh``
        gate (used by Gemma2 / Gemma3 style models).  Set to ``0.0`` to
        disable.
    v_head_dim
        Per-head dimension of the value projection.  Defaults to ``head_dim``
        (i.e. standard square QKV).
    sliding_window_size
        Sliding-window attention span.  ``-1`` means full context (no window).
    is_cross_attention
        ``True`` for cross-attention layers in encoder-decoder models.
    attn_type
        One of :class:`AttentionType`.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        sliding_window_size: int = -1,
        is_cross_attention: bool = False,
        attn_type: AttentionType = AttentionType.DECODER,
    ):
        super().__init__()

        self.tp_q_head_num: int = num_heads
        self.tp_k_head_num: int = num_kv_heads
        self.tp_v_head_num: int = num_kv_heads

        self.head_dim: int = head_dim
        self.qk_head_dim: int = head_dim
        self.v_head_dim: int = v_head_dim if v_head_dim != -1 else head_dim

        self.scaling: float = scaling
        self.layer_id: int = layer_id
        self.logit_cap: float = logit_cap
        self.sliding_window_size: int = (
            sliding_window_size if sliding_window_size is not None else -1
        )
        self.is_cross_attention: bool = is_cross_attention
        self.attn_type: AttentionType = attn_type

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Run attention for one batch.

        Parameters
        ----------
        q
            Query tensor, shape ``[num_tokens, tp_q_head_num * head_dim]``
            (or already reshaped to ``[num_tokens, tp_q_head_num, head_dim]``).
        k
            Key tensor, same leading dimension as ``q``, shape
            ``[num_tokens, tp_k_head_num * qk_head_dim]``.
            Pass ``None`` for cross-layer KV sharing (``v`` must also be
            ``None`` in this case).
        v
            Value tensor, shape
            ``[num_tokens, tp_v_head_num * v_head_dim]``.
        forward_batch
            Batch metadata and references to memory pools / backend.
        save_kv_cache
            When ``False``, skip writing K/V into the pool (useful for draft
            models in speculative decoding).
        **kwargs
            Passed through to the backend (e.g. ``q_rope``, ``k_rope``).
        """
        if k is not None:
            assert v is not None, "k and v must both be provided or both be None"
            k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
            v = v.view(-1, self.tp_v_head_num, self.v_head_dim)

        return forward_batch.attn_backend.forward(
            q, k, v, self, forward_batch, save_kv_cache, **kwargs
        )

    def extra_repr(self) -> str:
        return (
            f"layer_id={self.layer_id}, "
            f"q_heads={self.tp_q_head_num}, "
            f"kv_heads={self.tp_k_head_num}, "
            f"head_dim={self.head_dim}, "
            f"v_head_dim={self.v_head_dim}, "
            f"scaling={self.scaling:.4f}, "
            f"logit_cap={self.logit_cap}, "
            f"sliding_window={self.sliding_window_size}, "
            f"attn_type={self.attn_type.value}"
        )
