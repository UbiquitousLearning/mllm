"""Inference-only Qwen3.5 model for pymllm.

Implements the hybrid attention architecture:
- **Full attention layers** (standard transformer with RoPE + output gate)
- **GDN linear attention layers** (Gated Delta Network, O(n) complexity)

Layers alternate: linear, attention, linear, attention, ... based on
``full_attention_interval`` in the config.

Supports:
- Dense (non-MoE) variant
- Vision-Language (multimodal) via inheritance from Qwen3VL

Adapted from sglang's ``qwen3_5.py``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pymllm.layers.attention.radix_attention import RadixAttention
from pymllm.layers.embedding import VocabParallelEmbedding
from pymllm.layers.gated_delta_net import GatedDeltaNet
from pymllm.layers.linear import Linear
from pymllm.layers.mlp import MLP
from pymllm.layers.rms_norm import GemmaRMSNorm, RMSNorm
from pymllm.layers.rope import apply_rope_pos_ids
from pymllm.layers.utils import set_weight_attrs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _get_text_config(config):
    """Extract the text sub-config from a multimodal config, or return as-is."""
    return getattr(config, "text_config", config)


def _get_layer_types(config) -> List[str]:
    """Return per-layer type list: 'attention' or 'linear_attention'."""
    if hasattr(config, "layers_block_type"):
        return config.layers_block_type
    # Compute from full_attention_interval
    interval = getattr(config, "full_attention_interval", 2)
    n_layers = config.num_hidden_layers
    types = []
    for i in range(n_layers):
        if (i + 1) % interval == 0:
            types.append("attention")
        else:
            types.append("linear_attention")
    return types


# ---------------------------------------------------------------------------
# Full Attention Layer (with output gate + QK norm)
# ---------------------------------------------------------------------------


class Qwen3_5FullAttention(nn.Module):
    """Standard multi-head attention with RoPE, QK-norm, and optional output gate."""

    def __init__(self, config, layer_id: int, quant_config=None, prefix: str = ""):
        super().__init__()
        tc = _get_text_config(config)
        self.hidden_size = tc.hidden_size
        self.num_heads = tc.num_attention_heads
        self.num_kv_heads = tc.num_key_value_heads
        self.head_dim = getattr(tc, "head_dim", self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.layer_id = layer_id

        # Output gate: Qwen3.5 doubles the Q projection and uses half as a
        # sigmoid gate on the attention output.
        self.attn_output_gate = getattr(tc, "attn_output_gate", True)

        if self.attn_output_gate:
            q_proj_size = self.q_size * 2  # Q + gate
        else:
            q_proj_size = self.q_size

        def _get_qm(suffix):
            if quant_config is None:
                return None
            return quant_config.get_quant_method(
                layer=None, prefix=f"{prefix}.{suffix}" if prefix else suffix,
            )

        self.q_proj = Linear(self.hidden_size, q_proj_size, bias=False, quant_method=_get_qm("q_proj"))
        self.k_proj = Linear(self.hidden_size, self.kv_size, bias=False, quant_method=_get_qm("k_proj"))
        self.v_proj = Linear(self.hidden_size, self.kv_size, bias=False, quant_method=_get_qm("v_proj"))
        self.o_proj = Linear(self.q_size, self.hidden_size, bias=False, quant_method=_get_qm("o_proj"))

        # QK normalization
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=tc.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=tc.rms_norm_eps)

        # RoPE config
        self.partial_rotary_factor = getattr(tc, "partial_rotary_factor", 1.0)
        rope_config = getattr(tc, "rope_parameters", None) or getattr(tc, "rope_scaling", None) or {}
        self.rope_theta = rope_config.get("rope_theta", getattr(tc, "rope_theta", 10000.0))
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)

        # RadixAttention layer — delegates to the pluggable attention backend
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: Any,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if self.attn_output_gate:
            # Split Q into actual Q and gate
            q_gate = q.view(seq_len, self.num_heads, self.head_dim * 2)
            q, gate = q_gate.chunk(2, dim=-1)
            q = q.reshape(seq_len, -1)
            gate = gate.reshape(seq_len, -1)

        # QK norm
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(seq_len, -1)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(seq_len, -1)

        # RoPE (inplace; rotary_dim handles partial rotation)
        q = q.view(seq_len, self.num_heads, self.head_dim)
        k = k.view(seq_len, self.num_kv_heads, self.head_dim)
        apply_rope_pos_ids(
            q, k, positions, inplace=True,
            rotary_dim=self.rotary_dim, rope_theta=self.rope_theta,
        )
        q = q.reshape(seq_len, -1)
        k = k.reshape(seq_len, -1)

        # Standard attention via RadixAttention → attn_backend
        attn_output = self.attn(q, k, v, forward_batch)

        # Output gate
        if self.attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate)

        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# Full Attention Decoder Layer
# ---------------------------------------------------------------------------


class Qwen3_5AttentionDecoderLayer(nn.Module):
    """Decoder layer with full attention + MLP."""

    def __init__(self, config, layer_id: int, quant_config=None, prefix: str = ""):
        super().__init__()
        tc = _get_text_config(config)
        self.self_attn = Qwen3_5FullAttention(
            config, layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn" if prefix else "self_attn",
        )
        self.mlp = MLP(
            hidden_size=tc.hidden_size,
            intermediate_size=tc.intermediate_size,
            activation=tc.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp" if prefix else "mlp",
        )
        self.input_layernorm = GemmaRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: Any,
    ):
        # Pre-norm + residual
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states, forward_batch)

        # Post-attention norm + residual
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# ---------------------------------------------------------------------------
# Linear Attention (GDN) Decoder Layer
# ---------------------------------------------------------------------------


class Qwen3_5LinearDecoderLayer(nn.Module):
    """Decoder layer with GDN linear attention + MLP."""

    def __init__(self, config, layer_id: int, gdn_layer_idx: int = 0,
                 quant_config=None, prefix: str = ""):
        super().__init__()
        tc = _get_text_config(config)
        self.linear_attn = GatedDeltaNet(
            hidden_size=tc.hidden_size,
            num_k_heads=getattr(tc, "linear_num_key_heads", 16),
            num_v_heads=getattr(tc, "linear_num_value_heads", 32),
            head_k_dim=getattr(tc, "linear_key_head_dim", 128),
            head_v_dim=getattr(tc, "linear_value_head_dim", 128),
            conv_kernel_size=getattr(tc, "linear_conv_kernel_dim", 4),
            layer_id=layer_id,
            gdn_layer_idx=gdn_layer_idx,
            rms_norm_eps=tc.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_attn" if prefix else "linear_attn",
        )
        self.mlp = MLP(
            hidden_size=tc.hidden_size,
            intermediate_size=tc.intermediate_size,
            activation=tc.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp" if prefix else "mlp",
        )
        self.input_layernorm = GemmaRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: Any,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.linear_attn(hidden_states, forward_batch)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# ---------------------------------------------------------------------------
# Layer type registry
# ---------------------------------------------------------------------------

_DECODER_LAYER_TYPES = {
    "attention": Qwen3_5AttentionDecoderLayer,
    "linear_attention": Qwen3_5LinearDecoderLayer,
}


# ---------------------------------------------------------------------------
# Qwen3.5 Language Model (dense variant)
# ---------------------------------------------------------------------------


class Qwen3_5ForCausalLM(nn.Module):
    """Qwen3.5 causal language model with hybrid attention.

    Alternates between full attention and GDN linear attention layers.
    Dense (non-MoE) variant.
    """

    def __init__(self, config, quant_config=None):
        super().__init__()
        tc = _get_text_config(config)
        self.config = tc
        self.quant_config = quant_config
        self.hidden_size = tc.hidden_size
        self.vocab_size = tc.vocab_size

        # Embedding
        self.embed_tokens = VocabParallelEmbedding(tc.vocab_size, tc.hidden_size)

        # Build hybrid decoder layers with sequential GDN indexing
        layer_types = _get_layer_types(tc)
        self.layer_types = layer_types
        self.layers = nn.ModuleList()
        gdn_count = 0
        self.full_attn_layer_ids = set()
        for idx in range(tc.num_hidden_layers):
            layer_type = layer_types[idx]
            layer_prefix = f"layers.{idx}"
            if layer_type == "linear_attention":
                self.layers.append(
                    Qwen3_5LinearDecoderLayer(
                        config, idx, gdn_layer_idx=gdn_count,
                        quant_config=quant_config, prefix=layer_prefix,
                    )
                )
                gdn_count += 1
            else:
                self.layers.append(
                    Qwen3_5AttentionDecoderLayer(
                        config, idx,
                        quant_config=quant_config, prefix=layer_prefix,
                    )
                )
                self.full_attn_layer_ids.add(idx)
        self.num_gdn_layers = gdn_count

        # Final norm
        self.norm = GemmaRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)

        logger.info(
            "Qwen3_5ForCausalLM: %d layers (%d attention + %d GDN)",
            tc.num_hidden_layers,
            len(self.full_attn_layer_ids),
            self.num_gdn_layers,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )

        # Final normalization
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load HuggingFace checkpoint weights with name remapping."""
        # Always enable gate/up stacking.
        # - Pre-fused checkpoints (our GPTQ script): gate_up_proj.* keys pass
        #   path-component matching unchanged and are loaded directly.
        # - Unfused checkpoints (e.g. llm-compressor / RedHatAI style):
        #   gate_proj.* / up_proj.* are sharded into gate_up_proj.* below.
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded: Set[str] = set()

        for name, weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "visual" in name:
                continue
            if "language_model" in name:
                name = name.replace("model.language_model.", "")
            if name.startswith("model."):
                name = name[len("model."):]
            # NOTE: do NOT strip .self_attn — pymllm keeps it as a submodule

            # Handle stacked params (gate_up_proj = gate_proj + up_proj).
            # Use path-component matching (".gate_proj." in ".name.") to avoid
            # "gate_proj" falsely matching inside "gate_up_proj".
            matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in f".{name}.":
                    continue
                if "mlp.experts" in name:
                    continue
                # Use a tentative name; only commit once we confirm it's shardable.
                stacked_name = name.replace(weight_name, param_name)
                # llm-compressor W4A16: packed int4 saved as "weight",
                # pymllm registers the parameter as "weight_packed".
                if stacked_name not in params_dict and stacked_name.endswith(".weight"):
                    alt = stacked_name[: -len(".weight")] + ".weight_packed"
                    if alt in params_dict:
                        stacked_name = alt
                if stacked_name not in params_dict:
                    continue
                param = params_dict[stacked_name]
                # gate_up_proj is a plain Linear — manually place each shard.
                # Guard: skip metadata tensors (e.g. weight_shape [2]) that
                # cannot be evenly halved along dim-0.
                output_dim = param.shape[0] // 2
                if weight.shape[0] != output_dim:
                    break  # not a shardable weight tensor; fall through to direct load
                name = stacked_name  # commit rename
                param.data[shard_id * output_dim : (shard_id + 1) * output_dim].copy_(
                    weight
                )
                matched = True
                break

            if not matched:
                # llm-compressor W4A16: packed int4 saved as "weight",
                # pymllm registers the parameter as "weight_packed".
                if name not in params_dict and name.endswith(".weight"):
                    alt = name[: -len(".weight")] + ".weight_packed"
                    if alt in params_dict:
                        name = alt
                if name not in params_dict:
                    continue
                param = params_dict[name]
                loader = getattr(param, "weight_loader", None)
                if loader is not None:
                    loader(param, weight)
                else:
                    # Squeeze conv1d weight from [C, 1, K] to [C, K]
                    if weight.dim() != param.dim():
                        weight = weight.squeeze()
                    param.data.copy_(weight)

            loaded.add(name)

        logger.info("Loaded %d parameter tensors for Qwen3_5ForCausalLM", len(loaded))
        return loaded


# ---------------------------------------------------------------------------
# Qwen3.5 Vision encoder
# ---------------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embedding to vision Q/K. Inputs are ``(seq, heads, dim)``."""
    orig_dtype = q.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(-2).float()  # (seq, 1, dim)
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

class Qwen3_5VisionRotaryEmbedding(nn.Module):
    """Simple rotary embedding used inside the vision encoder."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)  # (seqlen, dim // 2)

class Qwen3_5VisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int,
                 hidden_act: str = "gelu_pytorch_tanh"):
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)
        if hidden_act == "gelu_pytorch_tanh":
            self.act = nn.GELU(approximate="tanh")
        elif hidden_act == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act(self.linear_fc1(x)))

class Qwen3_5VisionPatchEmbed(nn.Module):
    """3-D conv patch embedding (same structure as Qwen3VL)."""

    def __init__(self, patch_size: int, temporal_patch_size: int,
                 in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel,
                              stride=kernel, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        x = x.view(-1, self.in_channels, self.temporal_patch_size,
                    self.patch_size, self.patch_size)
        return self.proj(x.to(dtype=target_dtype)).view(-1, self.embed_dim)

class Qwen3_5VisionPatchMerger(nn.Module):
    """Merge ``spatial_merge_size²`` adjacent patches into one token.

    Unlike Qwen3VL there is **no deepstack** — a single merger outputs
    ``out_hidden_size`` features.
    """

    def __init__(self, hidden_size: int, spatial_merge_size: int,
                 out_hidden_size: int):
        super().__init__()
        self.merged_dim = hidden_size * (spatial_merge_size ** 2)
        # Norm on per-token hidden_size (before spatial concat)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merged_dim, self.merged_dim)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.merged_dim, out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (total_tokens, hidden_size), tokens already in merge order
        x = self.norm(x).view(-1, self.merged_dim)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))

class Qwen3_5VisionAttention(nn.Module):
    """Multi-head attention for the vision encoder.

    Processes each image/frame separately via ``cu_seqlens`` boundaries.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)  # each (seq, heads, dim)

        q, k = _apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Per-image/frame attention (no cross-image attention)
        outputs: List[torch.Tensor] = []
        for i in range(len(cu_seqlens) - 1):
            s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            qi = q[s:e].transpose(0, 1).unsqueeze(0)
            ki = k[s:e].transpose(0, 1).unsqueeze(0)
            vi = v[s:e].transpose(0, 1).unsqueeze(0)
            oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=False)
            outputs.append(oi.squeeze(0).transpose(0, 1))

        attn_out = torch.cat(outputs, dim=0).reshape(seq_len, -1).contiguous()
        return self.proj(attn_out)

class Qwen3_5VisionBlock(nn.Module):
    """Transformer block for the vision encoder."""

    def __init__(self, hidden_size: int, num_heads: int,
                 intermediate_size: int, hidden_act: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Qwen3_5VisionAttention(hidden_size, num_heads)
        self.mlp = Qwen3_5VisionMLP(hidden_size, intermediate_size, hidden_act)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cu_seqlens, cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x

class Qwen3_5VisionModel(nn.Module):
    """Qwen3.5 native vision encoder.

    Key differences from Qwen3VL's ``Qwen3VLVisionModel``:

    * **No deepstack** — single merger produces ``out_hidden_size`` features.
    * **Learned absolute position embedding** (``nn.Embedding``) with bilinear
      interpolation to arbitrary resolutions.
    * **2-D rotary position embedding** (row, col only — no temporal dim in
      the rotary frequencies).
    * Returns a **list of per-image tensors** instead of a single concatenated
      tensor with deepstack channels.
    """

    def __init__(
        self,
        depth: int = 27,
        hidden_size: int = 1152,
        hidden_act: str = "gelu_pytorch_tanh",
        intermediate_size: int = 4304,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 3584,
        num_position_embeddings: int = 2304,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.out_hidden_size = out_hidden_size

        self.patch_embed = Qwen3_5VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        # Learned absolute position embedding
        self.pos_embed = nn.Embedding(num_position_embeddings, hidden_size)
        self.num_grid_per_side = int(num_position_embeddings ** 0.5)

        # 2-D rotary embedding for ViT attention
        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([
            Qwen3_5VisionBlock(hidden_size, num_heads, intermediate_size, hidden_act)
            for _ in range(depth)
        ])

        self.merger = Qwen3_5VisionPatchMerger(
            hidden_size=hidden_size,
            spatial_merge_size=spatial_merge_size,
            out_hidden_size=out_hidden_size,
        )

    def _rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute 2-D rotary position embeddings for all patches."""
        merge = self.spatial_merge_size
        grid_list = grid_thw.tolist()
        device = self.rotary_pos_emb.inv_freq.device

        max_hw = max(max(int(h), int(w)) for _, h, w in grid_list)
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, head_dim//4)

        total = sum(int(t) * int(h) * int(w) for t, h, w in grid_list)
        pos_ids = torch.empty((total, 2), dtype=torch.long, device=device)

        offset = 0
        for nf, height, width in grid_list:
            nf, height, width = int(nf), int(height), int(width)
            mh, mw = height // merge, width // merge

            br = torch.arange(mh, device=device)
            bc = torch.arange(mw, device=device)
            ir = torch.arange(merge, device=device)
            ic = torch.arange(merge, device=device)

            row = (br[:, None, None, None] * merge + ir[None, None, :, None])
            col = (bc[None, :, None, None] * merge + ic[None, None, None, :])
            row = row.expand(mh, mw, merge, merge).reshape(-1)
            col = col.expand(mh, mw, merge, merge).reshape(-1)
            coords = torch.stack((row, col), dim=-1)
            if nf > 1:
                coords = coords.repeat(nf, 1)

            n = coords.shape[0]
            pos_ids[offset:offset + n] = coords
            offset += n

        embs = freq_table[pos_ids]       # (total, 2, head_dim//4)
        return embs.flatten(1)            # (total, head_dim//2)

    def _fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Bilinear-interpolate learned position embeddings to any resolution."""
        grid_list = grid_thw.tolist()
        ts = [int(r[0]) for r in grid_list]
        hs = [int(r[1]) for r in grid_list]
        ws = [int(r[2]) for r in grid_list]
        device = self.pos_embed.weight.device
        N = self.num_grid_per_side

        idx4: List[List[int]] = [[] for _ in range(4)]
        wt4: List[List[float]] = [[] for _ in range(4)]

        for h, w in zip(hs, ws):
            hi = torch.linspace(0, N - 1, h)
            wi = torch.linspace(0, N - 1, w)
            hf, wf = hi.int(), wi.int()
            hc = (hf + 1).clip(max=N - 1)
            wc = (wf + 1).clip(max=N - 1)
            dh, dw = hi - hf.float(), wi - wf.float()
            bh, bhc = hf * N, hc * N

            indices = [
                (bh[:, None] + wf[None, :]).flatten(),
                (bh[:, None] + wc[None, :]).flatten(),
                (bhc[:, None] + wf[None, :]).flatten(),
                (bhc[:, None] + wc[None, :]).flatten(),
            ]
            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
                ((1 - dh)[:, None] * dw[None, :]).flatten(),
                (dh[:, None] * (1 - dw)[None, :]).flatten(),
                (dh[:, None] * dw[None, :]).flatten(),
            ]
            for c in range(4):
                idx4[c].extend(indices[c].tolist())
                wt4[c].extend(weights[c].tolist())

        idx_t = torch.tensor(idx4, dtype=torch.long, device=device)
        wt_t = torch.tensor(wt4, dtype=self.pos_embed.weight.dtype, device=device)
        pe = self.pos_embed(idx_t) * wt_t[:, :, None]
        patch_pe = pe[0] + pe[1] + pe[2] + pe[3]

        # Split by image, rearrange to spatial-merge order
        parts = patch_pe.split([h * w for h, w in zip(hs, ws)])
        merge = self.spatial_merge_size
        result: List[torch.Tensor] = []
        for p, t, h, w in zip(parts, ts, hs, ws):
            p = p.repeat(t, 1)
            p = (p.view(t, h // merge, merge, w // merge, merge, -1)
                  .permute(0, 1, 3, 2, 4, 5)
                  .flatten(0, 4))
            result.append(p)
        return torch.cat(result)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Run vision encoder.

        Returns a **list** of per-image tensors, each of shape
        ``[num_merged_tokens_i, out_hidden_size]``.
        """
        hidden = self.patch_embed(pixel_values)

        # Add learned absolute position embeddings
        hidden = hidden + self._fast_pos_embed_interpolate(grid_thw)

        # Rotary embeddings for ViT attention
        rot = self._rot_pos_emb(grid_thw)
        emb = torch.cat((rot, rot), dim=-1)
        cos, sin = emb.cos(), emb.sin()

        # Per-frame cumulative sequence lengths for attention
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden = blk(hidden, cu_seqlens, cos, sin)

        # Spatial merge → project to LLM hidden size
        merged = self.merger(hidden)

        # Split into per-image tensors
        split_sizes = (grid_thw.prod(-1) // (self.spatial_merge_size ** 2)).tolist()
        return list(torch.split(merged, split_sizes))


def get_rope_index_qwen3_5(
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor],
    image_token_id: int,
    vision_start_token_id: int,
    spatial_merge_size: int,
    start_pos: int = 0,
) -> Tuple[torch.Tensor, int]:
    """Compute M-RoPE 3-D position IDs for a single Qwen3.5 sequence.

    Compatible with pymllm's ``[3, T]`` M-RoPE interface.

    Key differences from Qwen3VL's ``get_rope_index``:

    * **Position advance** for vision tokens is
      ``max(H, W) // spatial_merge_size``
      (Qwen3VL uses ``max(T, H, W)``).
    * **Temporal position** for image tokens is constant (``= current_pos``).

    Parameters
    ----------
    start_pos
        Starting M-RoPE position counter.  Must equal ``extend_prefix_lens``
        so that positions are computed correctly when part of the prompt is
        already cached (radix-cache prefix hit).  Returns ``delta`` adjusted
        to subtract ``start_pos`` so that the decode formula
        ``pos = (seq_len - 1) + delta`` remains correct.
    """
    total = input_ids.shape[0]
    device = input_ids.device
    pos = torch.zeros(3, total, dtype=torch.long, device=device)

    if image_grid_thw is None or image_grid_thw.shape[0] == 0:
        arange = torch.arange(total, dtype=torch.long, device=device) + start_pos
        pos[0] = arange
        pos[1] = arange
        pos[2] = arange
        # delta adjusted: (start_pos + total) - total - start_pos = 0
        return pos, 0

    ids = input_ids.cpu().tolist()
    grids = image_grid_thw.cpu().tolist()

    cur = start_pos
    img_idx = 0
    i = 0

    while i < total:
        if ids[i] == vision_start_token_id and img_idx < len(grids):
            # vision_start token → regular sequential position
            pos[:, i] = cur
            cur += 1
            i += 1

            t_g = int(grids[img_idx][0])
            h_g = int(grids[img_idx][1])
            w_g = int(grids[img_idx][2])
            gh = h_g // spatial_merge_size
            gw = w_g // spatial_merge_size
            n_img = t_g * gh * gw

            # Temporal: constant
            t_pos = torch.full((n_img,), cur, device=device, dtype=torch.long)
            # Height
            h_pos = (torch.arange(gh, device=device)
                     .repeat_interleave(gw).repeat(t_g) + cur)
            # Width
            w_pos = (torch.arange(gw, device=device)
                     .repeat(gh * t_g) + cur)

            pos[0, i:i + n_img] = t_pos
            pos[1, i:i + n_img] = h_pos
            pos[2, i:i + n_img] = w_pos

            cur += max(gh, gw)
            i += n_img
            img_idx += 1
        else:
            pos[:, i] = cur
            cur += 1
            i += 1

    # delta = (cur - start_pos) - total, so that decode uses:
    #   pos_1d = (seq_len - 1) + delta
    #          = (prefix + total + 1 - 1) + (cur - start_pos - total)
    #          = cur  (the correct next M-RoPE position)
    return pos, (cur - start_pos) - total


# ---------------------------------------------------------------------------
# Qwen3.5 Vision-Language Model
# ---------------------------------------------------------------------------


class Qwen3_5ForConditionalGeneration(nn.Module):
    """
    Qwen3.5 multimodal model (text + vision).
    """

    def __init__(self, config, quant_config=None):
        super().__init__()

        self.config = config
        self.quant_config = quant_config
        tc = _get_text_config(config)

        # Vision encoder — NOT quantized
        vision_config = getattr(config, "vision_config", None)
        self.spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
        if vision_config is not None:
            self.visual = Qwen3_5VisionModel(
                depth=getattr(vision_config, "depth", 27),
                hidden_size=getattr(vision_config, "hidden_size", 1152),
                hidden_act=getattr(vision_config, "hidden_act", "gelu_pytorch_tanh"),
                intermediate_size=getattr(vision_config, "intermediate_size", 4304),
                num_heads=getattr(vision_config, "num_heads", 16),
                in_channels=getattr(vision_config, "in_channels", 3),
                patch_size=getattr(vision_config, "patch_size", 16),
                spatial_merge_size=self.spatial_merge_size,
                temporal_patch_size=getattr(vision_config, "temporal_patch_size", 2),
                out_hidden_size=getattr(vision_config, "out_hidden_size", tc.hidden_size),
                num_position_embeddings=getattr(
                    vision_config, "num_position_embeddings", 2304
                ),
            )
        else:
            self.visual = None

        # Language model
        self.model = Qwen3_5ForCausalLM(config, quant_config=quant_config)

        # Expose hybrid model metadata for ModelRunner
        self.num_gdn_layers = self.model.num_gdn_layers
        self.full_attn_layer_ids = self.model.full_attn_layer_ids

        # LM head
        self.lm_head = Linear(tc.hidden_size, tc.vocab_size, bias=False)
        if getattr(tc, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

        # Vision token IDs
        self.image_token_id = getattr(config, "image_token_id", 248056)
        self.video_token_id = getattr(config, "video_token_id", 248057)
        self.vision_start_token_id = getattr(config, "vision_start_token_id", 248053)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Process vision inputs if provided
        pixel_values = (
            pixel_values if pixel_values is not None
            else getattr(forward_batch, "pixel_values", None)
        )
        image_grid_thw = (
            image_grid_thw if image_grid_thw is not None
            else getattr(forward_batch, "image_grid_thw", None)
        )

        if forward_batch.forward_mode.is_extend():
            # Prefill: compute per-sequence 3-D position IDs from input_ids
            # and image grids, then store per-request deltas for future decode.
            mrope_positions_list: List[torch.Tensor] = []
            deltas: List[int] = []
            image_idx_offset = 0

            for i in range(forward_batch.batch_size):
                start = int(forward_batch.extend_start_loc[i].item())
                length = int(forward_batch.extend_seq_lens[i].item())
                seq_ids = input_ids[start : start + length]

                # Prefix length (cached tokens not in input_ids).
                # Positions must start from prefix_len so that new tokens
                # get correct M-RoPE positions even on radix cache hits.
                prefix_len = int(forward_batch.extend_prefix_lens[i].item()) \
                    if forward_batch.extend_prefix_lens is not None else 0

                # Determine how many images belong to this sequence.
                num_img = int((seq_ids == self.vision_start_token_id).sum().item())
                if image_grid_thw is not None and num_img > 0:
                    thw_seq = image_grid_thw[
                        image_idx_offset : image_idx_offset + num_img
                    ]
                    image_idx_offset += num_img
                else:
                    thw_seq = None

                pos3d, delta = get_rope_index_qwen3_5(
                    seq_ids,
                    thw_seq,
                    self.image_token_id,
                    self.vision_start_token_id,
                    self.spatial_merge_size,
                    start_pos=prefix_len,
                )
                mrope_positions_list.append(pos3d)
                deltas.append(delta)

            # Concatenate across sequences: [3, total_extend_tokens]
            positions = torch.cat(mrope_positions_list, dim=1).contiguous()
            forward_batch.mrope_position_deltas = torch.tensor(
                deltas, dtype=torch.int64, device=input_ids.device
            )
        else:
            # Decode: each sequence emits exactly one token.  Apply the stored
            # per-request delta so the position matches the image extent.
            stored_deltas = getattr(forward_batch, "mrope_position_deltas", None)
            if stored_deltas is not None:
                pos_1d = forward_batch.positions + stored_deltas
            else:
                pos_1d = forward_batch.positions
            positions = pos_1d.unsqueeze(0).expand(3, -1).contiguous()  # [3, batch_size]

        if (
            pixel_values is not None
            and image_grid_thw is not None
            and self.visual is not None
            and not forward_batch.forward_mode.is_decode()
        ):
            input_embeds = self.model.embed_tokens(input_ids)
            # Vision encoder → list of per-image tensors
            image_embeds_list = self.visual(pixel_values, grid_thw=image_grid_thw)
            image_embeds = torch.cat(image_embeds_list, dim=0)

            # Replace image-token positions with visual embeddings
            image_mask = input_ids == self.image_token_id
            n_holders = image_mask.sum().item()
            n_embeds = image_embeds.shape[0]
            if n_holders != n_embeds:
                logger.warning(
                    "image placeholder count (%d) != vision embed count (%d)",
                    n_holders, n_embeds,
                )
            if image_mask.any():
                mask_3d = image_mask.unsqueeze(-1).expand_as(input_embeds)
                input_embeds = input_embeds.masked_scatter(mask_3d, image_embeds)

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )

        # LM head
        logits = self.lm_head(hidden_states)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load HuggingFace checkpoint weights (visual + language)."""
        visual_weights: List[Tuple[str, torch.Tensor]] = []
        language_weights: List[Tuple[str, torch.Tensor]] = []

        for name, weight in weights:
            if "visual" in name or "model.visual" in name:
                name = name.replace("model.visual.", "visual.")
                visual_weights.append((name, weight))
            else:
                language_weights.append((name, weight))

        # Language
        self.model.load_weights(language_weights)

        # Vision
        if self.visual is not None and visual_weights:
            params_dict = dict(self.named_parameters())
            loaded = 0
            for name, weight in visual_weights:
                if name not in params_dict:
                    logger.debug("Skipping visual weight: %s", name)
                    continue
                param = params_dict[name]
                loader = getattr(param, "weight_loader", None)
                if loader is not None:
                    loader(param, weight)
                else:
                    if weight.dim() != param.dim():
                        weight = weight.squeeze()
                    param.data.copy_(weight)
                loaded += 1
            logger.info("Loaded %d visual weight tensors", loaded)

        logger.info("Qwen3_5ForConditionalGeneration weights loaded")
