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

    def __init__(self, config, layer_id: int):
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

        self.q_proj = Linear(self.hidden_size, q_proj_size, bias=False)
        self.k_proj = Linear(self.hidden_size, self.kv_size, bias=False)
        self.v_proj = Linear(self.hidden_size, self.kv_size, bias=False)
        self.o_proj = Linear(self.q_size, self.hidden_size, bias=False)

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

    def __init__(self, config, layer_id: int):
        super().__init__()
        tc = _get_text_config(config)
        self.self_attn = Qwen3_5FullAttention(config, layer_id)
        self.mlp = MLP(
            hidden_size=tc.hidden_size,
            intermediate_size=tc.intermediate_size,
            activation=tc.hidden_act,
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

    def __init__(self, config, layer_id: int, gdn_layer_idx: int = 0):
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
        )
        self.mlp = MLP(
            hidden_size=tc.hidden_size,
            intermediate_size=tc.intermediate_size,
            activation=tc.hidden_act,
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

    def __init__(self, config):
        super().__init__()
        tc = _get_text_config(config)
        self.config = tc
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
            if layer_type == "linear_attention":
                self.layers.append(
                    Qwen3_5LinearDecoderLayer(config, idx, gdn_layer_idx=gdn_count)
                )
                gdn_count += 1
            else:
                self.layers.append(
                    Qwen3_5AttentionDecoderLayer(config, idx)
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

            # Handle stacked params (gate_up_proj = gate_proj + up_proj)
            matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                # gate_up_proj is a plain Linear — manually place each shard
                output_dim = param.shape[0] // 2
                param.data[shard_id * output_dim : (shard_id + 1) * output_dim].copy_(
                    weight
                )
                matched = True
                break

            if not matched:
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
# Qwen3.5 Vision-Language Model
# ---------------------------------------------------------------------------


class Qwen3_5ForConditionalGeneration(nn.Module):
    """Qwen3.5 multimodal model (text + vision).

    Inherits vision encoder from Qwen3VL and uses Qwen3.5's hybrid
    language model.
    """

    def __init__(self, config):
        super().__init__()
        from pymllm.models.qwen3_vl import (
            Qwen3VLVisionModel,
        )

        self.config = config
        tc = _get_text_config(config)

        # Vision encoder (reuse Qwen3VL's vision model)
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            self.visual = Qwen3VLVisionModel(
                depth=getattr(vision_config, "depth", 27),
                hidden_size=getattr(vision_config, "hidden_size", 1152),
                hidden_act=getattr(vision_config, "hidden_act", "gelu_pytorch_tanh"),
                intermediate_size=getattr(vision_config, "intermediate_size", 4304),
                num_heads=getattr(vision_config, "num_heads", 16),
                in_channels=getattr(vision_config, "in_channels", 3),
                patch_size=getattr(vision_config, "patch_size", 16),
                spatial_merge_size=getattr(vision_config, "spatial_merge_size", 2),
                temporal_patch_size=getattr(vision_config, "temporal_patch_size", 2),
                out_hidden_size=getattr(vision_config, "out_hidden_size", 3584),
                num_position_embeddings=getattr(
                    vision_config, "num_position_embeddings", 2304
                ),
                deepstack_visual_indexes=getattr(
                    vision_config, "deepstack_visual_indexes", [8, 16, 24]
                ),
                norm_eps=getattr(tc, "rms_norm_eps", 1e-6),
            )
        else:
            self.visual = None

        # Language model
        self.model = Qwen3_5ForCausalLM(config)

        # Expose hybrid model metadata for ModelRunner
        self.num_gdn_layers = self.model.num_gdn_layers
        self.full_attn_layer_ids = self.model.full_attn_layer_ids

        # LM head (tied to embedding when tie_word_embeddings=True)
        self.lm_head = Linear(tc.hidden_size, tc.vocab_size, bias=False)
        if getattr(tc, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

        # Vision token IDs
        self.image_token_id = getattr(config, "image_token_id", 151655)
        self.video_token_id = getattr(config, "video_token_id", 151656)

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
        if input_embeds is None and pixel_values is not None and self.visual is not None:
            input_embeds = self.model.embed_tokens(input_ids)
            # Run vision encoder
            visual_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            # Replace image/video token positions with visual embeddings
            mask = (input_ids == self.image_token_id) | (input_ids == self.video_token_id)
            if mask.any():
                input_embeds[mask] = visual_embeds.reshape(-1, visual_embeds.shape[-1])

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
        """Load weights, dispatching visual vs language params."""
        visual_weights = []
        language_weights = []

        for name, weight in weights:
            if "visual" in name or "model.visual" in name:
                # Normalize visual weight names
                name = name.replace("model.visual.", "visual.")
                name = name.replace("attn.qkv.", "attn.qkv_proj.")
                visual_weights.append((name, weight))
            else:
                language_weights.append((name, weight))

        # Load language model weights
        self.model.load_weights(language_weights)

        # Load visual weights
        if self.visual is not None and visual_weights:
            params_dict = dict(self.named_parameters())
            for name, weight in visual_weights:
                if name in params_dict:
                    param = params_dict[name]
                    loader = getattr(param, "weight_loader", None)
                    if loader is not None:
                        loader(param, weight)
                    else:
                        param.data.copy_(weight)

        logger.info("Qwen3_5ForConditionalGeneration weights loaded")
