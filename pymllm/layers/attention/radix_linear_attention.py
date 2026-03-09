"""RadixLinearAttention -- GDN linear-attention layer for hybrid models.

Analogous to :class:`RadixAttention` but for GDN (Gated Delta Net) layers.
Stores per-layer GDN parameters and delegates computation to the
:meth:`AttentionBackend.forward_gdn` method on the current
:class:`~pymllm.engine.forward_batch.ForwardBatch`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from pymllm.engine.forward_batch import ForwardBatch


class RadixLinearAttention(nn.Module):
    """GDN linear-attention layer that delegates to the attention backend.

    Each GDN layer in a pymllm model creates one ``RadixLinearAttention``
    with a unique ``layer_id`` and ``gdn_layer_idx``.  During forward, it
    calls ``forward_batch.attn_backend.forward_gdn(...)`` which routes to
    the appropriate GDN backend implementation.

    Parameters
    ----------
    layer_id : int
        Global zero-based layer index within the model.
    gdn_layer_idx : int
        Sequential zero-based index among GDN layers only (not global).
        Used to index into :class:`~pymllm.mem_cache.memory_pool.GDNPool`.
    num_k_heads : int
        Number of key heads.
    num_v_heads : int
        Number of value heads.
    head_k_dim : int
        Per-head key dimension.
    head_v_dim : int
        Per-head value dimension.
    conv_weight : nn.Parameter
        Reference to the GDNConv1d weight parameter.
    A_log : nn.Parameter
        Log-space decay parameter.
    dt_bias : nn.Parameter
        Bias for the decay gate.
    """

    def __init__(
        self,
        layer_id: int,
        gdn_layer_idx: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_weight: nn.Parameter,
        A_log: nn.Parameter,
        dt_bias: nn.Parameter,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.gdn_layer_idx = gdn_layer_idx
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        # Store references to model parameters (not copies)
        self.conv_weight = conv_weight
        self.A_log = A_log
        self.dt_bias = dt_bias

    def forward(
        self,
        forward_batch: "ForwardBatch",
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate GDN computation to the attention backend.

        Parameters
        ----------
        forward_batch
            Batch metadata with ``attn_backend`` attached.
        mixed_qkv
            Concatenated Q/K/V projection output before conv1d.
        a
            Decay gate input, shape ``[num_tokens, num_v_heads]``.
        b
            Update gate input, shape ``[num_tokens, num_v_heads]``.

        Returns
        -------
        torch.Tensor
            GDN attention output, shape ``[num_tokens, num_v_heads * head_v_dim]``.
        """
        return forward_batch.attn_backend.forward_gdn(
            layer=self,
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
        )

    def extra_repr(self) -> str:
        return (
            f"layer_id={self.layer_id}, "
            f"gdn_layer_idx={self.gdn_layer_idx}, "
            f"k_heads={self.num_k_heads}, "
            f"v_heads={self.num_v_heads}, "
            f"k_dim={self.head_k_dim}, "
            f"v_dim={self.head_v_dim}"
        )
