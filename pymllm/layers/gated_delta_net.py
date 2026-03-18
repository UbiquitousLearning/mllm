"""Gated Delta Network (GDN) linear attention for Qwen3.5.

This implements the linear attention mechanism used in Qwen3.5's hybrid
architecture.  GDN alternates with standard full-attention layers.

Core formulation (decode, per-head):
    g_t = -exp(A_log) * softplus(a_t + dt_bias)
    beta_t = sigmoid(b_t)
    state_t = exp(g_t) * state_{t-1} + beta_t * (k_t outer v_t)
    output_t = (q_t @ state_t)

State is externalized into a :class:`~pymllm.mem_cache.memory_pool.GDNPool`
and computation is delegated to the attention backend via
:class:`~pymllm.layers.attention.radix_linear_attention.RadixLinearAttention`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from pymllm.layers.base import MllmBaseLayer
from pymllm.layers.linear import Linear
from pymllm.layers.utils import set_weight_attrs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conv1d weight holder
# ---------------------------------------------------------------------------


class GDNConv1d(nn.Module):
    """Causal 1D convolution weight holder for GDN sequence mixing.

    The actual convolution computation is performed by the GDN backend
    using pooled conv states.  This module only holds the learnable weight.
    """

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))


# ---------------------------------------------------------------------------
# GatedDeltaNet — main GDN layer
# ---------------------------------------------------------------------------


class GatedDeltaNet(MllmBaseLayer):
    """Gated Delta Network linear attention layer for Qwen3.5.

    State is externalized into a GDNPool and computation is delegated to
    the attention backend via RadixLinearAttention.

    Parameters
    ----------
    hidden_size : int
        Model hidden dimension.
    num_k_heads : int
        Number of key heads.
    num_v_heads : int
        Number of value heads.
    head_k_dim : int
        Per-head key dimension.
    head_v_dim : int
        Per-head value dimension.
    conv_kernel_size : int
        Causal conv1d kernel width.
    layer_id : int
        Global layer index.
    gdn_layer_idx : int
        Sequential index among GDN layers (0-based).
    rms_norm_eps : float
        Epsilon for gated RMS normalization.
    """

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int = 16,
        num_v_heads: int = 32,
        head_k_dim: int = 128,
        head_v_dim: int = 128,
        conv_kernel_size: int = 4,
        layer_id: int = 0,
        gdn_layer_idx: int = 0,
        rms_norm_eps: float = 1e-6,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = head_k_dim * num_k_heads
        self.value_dim = head_v_dim * num_v_heads
        self.conv_kernel_size = conv_kernel_size
        self.layer_id = layer_id
        self.gdn_layer_idx = gdn_layer_idx

        def _get_qm(suffix, out_features):
            # Skip quantization for small projections — Marlin kernels
            # require minimum thread tile sizes that exceed these dims.
            if quant_config is None or out_features < 64:
                return None
            return quant_config.get_quant_method(
                layer=None, prefix=f"{prefix}.{suffix}" if prefix else suffix,
            )

        # Input projections
        self.in_proj_qkv = Linear(
            hidden_size, self.key_dim * 2 + self.value_dim, bias=False,
            quant_method=_get_qm("in_proj_qkv", self.key_dim * 2 + self.value_dim),
        )
        self.in_proj_z = Linear(
            hidden_size, self.value_dim, bias=False,
            quant_method=_get_qm("in_proj_z", self.value_dim),
        )
        self.in_proj_a = Linear(
            hidden_size, num_v_heads, bias=False,
            quant_method=_get_qm("in_proj_a", num_v_heads),
        )
        self.in_proj_b = Linear(
            hidden_size, num_v_heads, bias=False,
            quant_method=_get_qm("in_proj_b", num_v_heads),
        )

        # Causal convolution (weight only — computation is in the backend)
        self.conv1d = GDNConv1d(self.key_dim * 2 + self.value_dim, conv_kernel_size)

        # State parameters (must stay float32 for numerical stability)
        self.A_log = nn.Parameter(torch.empty(num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.ones(num_v_heads, dtype=torch.float32))
        set_weight_attrs(self.A_log, {"weight_loader": self.weight_loader})
        set_weight_attrs(self.dt_bias, {"weight_loader": self.weight_loader})

        # Gated RMSNorm (mllm-kernel accelerated)
        from pymllm.layers.rms_norm_gated import RMSNormGated
        self.norm = RMSNormGated(head_v_dim, eps=rms_norm_eps, norm_before_gate=True)

        # Output projection
        self.out_proj = Linear(
            self.value_dim, hidden_size, bias=False,
            quant_method=_get_qm("out_proj", hidden_size),
        )

        # RadixLinearAttention — delegates to the attention backend
        from pymllm.layers.attention.radix_linear_attention import RadixLinearAttention
        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            gdn_layer_idx=gdn_layer_idx,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_weight=self.conv1d.weight,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

    def forward(
        self, hidden_states: torch.Tensor, forward_batch: Any = None,
    ) -> torch.Tensor:
        seq_len, _ = hidden_states.shape

        # Input projections
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        a = self.in_proj_a(hidden_states)
        b = self.in_proj_b(hidden_states)

        # Delegate to backend via RadixLinearAttention
        # The backend handles: conv1d, SiLU, split, gating, recurrent update
        attn_out = self.attn(forward_batch, mixed_qkv, a, b)

        # Gated norm + output projection
        attn_out = attn_out.view(seq_len, self.num_v_heads, self.head_v_dim)
        z = z.view(seq_len, self.num_v_heads, self.head_v_dim)

        attn_flat = attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        normed = self.norm(attn_flat, z_flat)
        normed = normed.view(seq_len, self.num_v_heads, self.head_v_dim)
        normed = normed.reshape(seq_len, self.value_dim)
        return self.out_proj(normed)
