# Copyright (c) MLLM Team.
# Licensed under the MIT License.
#
# Qwen3.5 QNN quantization model.
#
# Full attention layers have QDQ nodes for QNN LPBQ quantization.
# GDN layers are standard (unquantized) — they stay on CPU at runtime.
# Based on the Qwen3 quantization model pattern.

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config as HFQwen3_5Config

# Import the original GDN layer (no quantization needed)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5DynamicCache,
)
from transformers.masking_utils import create_causal_mask

# QDQ and quantized modules
from pymllm.mobile.backends.qualcomm.transformers.core.rms_norm import QRMSNorm
from pymllm.mobile.backends.qualcomm.transformers.core.qlinear import QLinearLPBQ
from pymllm.mobile.backends.qualcomm.transformers.core.qdq import (
    ActivationQDQ,
    FixedActivationQDQ,
)
from pymllm.mobile.backends.qualcomm.transformers.core.embedding import QEmbedding
from pymllm.mobile.backends.qualcomm.transformers.core.observer import ConcatObserver


# ============================================================================
# MLP with QDQ nodes (same pattern as Qwen3)
# ============================================================================

class Qwen3_5MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = QLinearLPBQ(self.hidden_size, self.intermediate_size, bias=False, block_size=16)
        self.up_proj = QLinearLPBQ(self.hidden_size, self.intermediate_size, bias=False, block_size=16)
        self.down_proj = QLinearLPBQ(self.intermediate_size, self.hidden_size, bias=False, block_size=16)

        # QDQ nodes
        self.up_proj_input_qdq = ActivationQDQ(bits=16)
        self.up_proj_output_qdq = ActivationQDQ(bits=16)
        self.gate_proj_output_qdq = ActivationQDQ(bits=16)
        self.act_output_qdq = ActivationQDQ(bits=16)
        self.down_proj_input_qdq = ActivationQDQ(bits=16)
        sigmoid_scale = 1.0 / (65535 - 0 + 1)
        self.sigmoid_output_qdq = FixedActivationQDQ(scale=sigmoid_scale, zero_point=0, bits=16)

    def forward(self, x):
        x = self.up_proj_input_qdq(x)
        up_result = self.up_proj_output_qdq(self.up_proj(x))
        gate_result = self.gate_proj_output_qdq(self.gate_proj(x))
        gate_result = self.act_output_qdq(gate_result * self.sigmoid_output_qdq(F.sigmoid(gate_result)))
        o = self.down_proj_input_qdq(gate_result * up_result)
        o = self.down_proj(o)
        return o


# ============================================================================
# Helpers
# ============================================================================

def rotate_half(x, x_observer=None, x2_neg_fake_quant=None, concat_observer=None):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    if x2_neg_fake_quant is not None and concat_observer is not None:
        return concat_observer(torch.cat((x2_neg_fake_quant(-x2), x1), dim=-1))
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply partial RoPE: only rotate first rotary_dim dims."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ============================================================================
# Full Attention with QDQ nodes + output gating + partial RoPE
# ============================================================================

class Qwen3_5Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        # Q proj is 2x wide for output gating
        self.q_proj = QLinearLPBQ(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias, block_size=16,
        )
        self.k_proj = QLinearLPBQ(
            config.hidden_size, config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias, block_size=16,
        )
        self.v_proj = QLinearLPBQ(
            config.hidden_size, config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias, block_size=16,
        )
        self.o_proj = QLinearLPBQ(
            config.num_attention_heads * self.head_dim, config.hidden_size,
            bias=config.attention_bias, block_size=16,
        )

        # GemmaRMSNorm for QK normalization (pre-baked +1.0 at deploy)
        self.q_norm = QRMSNorm(self.head_dim, eps=config.rms_norm_eps, quant_bits=16)
        self.k_norm = QRMSNorm(self.head_dim, eps=config.rms_norm_eps, quant_bits=16)

        # QDQ nodes for attention
        self.q_proj_input_qdq = ActivationQDQ(bits=16)
        self.q_norm_input_qdq = ActivationQDQ(bits=16)
        self.q_norm_output_qdq = ActivationQDQ(bits=16)
        self.k_norm_input_qdq = ActivationQDQ(bits=16)
        self.k_norm_output_qdq = ActivationQDQ(bits=16)

        # Partial RoPE QDQ (only on rotary_dim subset)
        self.q_rope_mul_0_output_qdq = ActivationQDQ(bits=16)
        self.q_rope_mul_1_output_qdq = ActivationQDQ(bits=16)
        self.q_rope_add_0_output_qdq = ActivationQDQ(bits=16)
        self.k_rope_mul_0_output_qdq = ActivationQDQ(bits=16)
        self.k_rope_mul_1_output_qdq = ActivationQDQ(bits=16)
        self.k_rope_add_0_output_qdq = ActivationQDQ(bits=16)

        # Concat observers for rotate_half
        self.q_rope_concat_observer = ConcatObserver(
            dtype=torch.int32, qscheme=torch.per_tensor_affine,
            reduce_range=False, quant_min=0, quant_max=2**16 - 1,
            eps=0.0001 / 65535, is_dynamic=False,
        )
        self.q_rope_neg_half_qdq = ActivationQDQ(bits=16)
        self.k_rope_concat_observer = ConcatObserver(
            dtype=torch.int32, qscheme=torch.per_tensor_affine,
            reduce_range=False, quant_min=0, quant_max=2**16 - 1,
            eps=0.0001 / 65535, is_dynamic=False,
        )
        self.k_rope_neg_half_qdq = ActivationQDQ(bits=16)

        self.k_rope_concat_observer.add_observer(self.k_norm_output_qdq.fake_quant.activation_post_process)
        self.k_rope_concat_observer.add_observer(self.k_rope_neg_half_qdq.fake_quant.activation_post_process)
        self.q_rope_concat_observer.add_observer(self.q_norm_output_qdq.fake_quant.activation_post_process)
        self.q_rope_concat_observer.add_observer(self.q_rope_neg_half_qdq.fake_quant.activation_post_process)

        # Pass-through QDQ for non-rotated dims
        self.q_pass_qdq = ActivationQDQ(bits=16)
        self.k_pass_qdq = ActivationQDQ(bits=16)
        self.q_rope_cat_output_qdq = ActivationQDQ(bits=16)
        self.k_rope_cat_output_qdq = ActivationQDQ(bits=16)

        # KV cache quantization
        self.k_cast_to_int8_qdq = ActivationQDQ(bits=8, qscheme=torch.per_tensor_symmetric)
        self.v_cast_to_int8_qdq = ActivationQDQ(bits=8, qscheme=torch.per_tensor_symmetric)
        self.v_cast_to_int16_qdq = ActivationQDQ(bits=16)

        # Attention computation QDQ
        self.qk_matmul_output_qdq = ActivationQDQ(bits=16)
        self.scaling_qdq = ActivationQDQ(bits=16)
        self.neg_20_qdq = ActivationQDQ(bits=16)
        self.reduce_min_output_qdq = ActivationQDQ(bits=16)
        self.mul_0_output_qdq = ActivationQDQ(bits=16)
        self.minus_0_output_qdq = ActivationQDQ(bits=16)
        self.softmax_output_qdq = ActivationQDQ(bits=16)
        self.attn_value_matmul_output_qdq = ActivationQDQ(bits=16)
        self.where_attn_qdq = ActivationQDQ(bits=16)

        # Output gating QDQ
        sigmoid_scale = 1.0 / (65535 + 1)
        self.attn_gate_sigmoid_qdq = FixedActivationQDQ(scale=sigmoid_scale, zero_point=0, bits=16)
        self.attn_gate_mul_qdq = ActivationQDQ(bits=16)
        self.gate_transpose_qdq = ActivationQDQ(bits=16)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        hidden_states = self.q_proj_input_qdq(hidden_states)

        # Q proj + split into query and gate
        q_raw = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2)
        query_states, gate = torch.chunk(q_raw, 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)  # [B, S, num_heads * head_dim]

        # QK normalization
        query_states = self.q_norm(
            self.q_norm_input_qdq(query_states.view(hidden_shape))
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_norm_input_qdq(self.k_proj(hidden_states)).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.q_norm_output_qdq(query_states)
        key_states = self.k_norm_output_qdq(key_states)

        # Partial RoPE: only rotate first rotary_dim dims
        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        rotary_dim = cos.shape[-1]
        q_rot = query_states[..., :rotary_dim]
        q_pass = query_states[..., rotary_dim:]
        k_rot = key_states[..., :rotary_dim]
        k_pass = key_states[..., rotary_dim:]

        # Apply RoPE to rotary part with QDQ
        q_rot_applied = self.q_rope_add_0_output_qdq(
            self.q_rope_mul_0_output_qdq(q_rot * cos)
            + self.q_rope_mul_1_output_qdq(
                rotate_half(
                    q_rot,
                    self.q_norm_output_qdq.fake_quant.activation_post_process,
                    self.q_rope_neg_half_qdq,
                    self.q_rope_concat_observer,
                ) * sin
            )
        )
        k_rot_applied = self.k_rope_add_0_output_qdq(
            self.k_rope_mul_0_output_qdq(k_rot * cos)
            + self.k_rope_mul_1_output_qdq(
                rotate_half(
                    k_rot,
                    self.k_norm_output_qdq.fake_quant.activation_post_process,
                    self.k_rope_neg_half_qdq,
                    self.k_rope_concat_observer,
                ) * sin
            )
        )

        # Concat rotated + pass-through
        query_states = self.q_rope_cat_output_qdq(
            torch.cat([q_rot_applied, self.q_pass_qdq(q_pass)], dim=-1)
        )
        key_states = self.k_rope_cat_output_qdq(
            torch.cat([k_rot_applied, self.k_pass_qdq(k_pass)], dim=-1)
        )

        # KV cache quantization
        key_states = self.k_cast_to_int8_qdq(key_states)
        value_states = self.v_cast_to_int8_qdq(self.v_cast_to_int16_qdq(value_states))

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Attention scores
        attn_weights = self.mul_0_output_qdq(
            self.qk_matmul_output_qdq(torch.matmul(query_states, key_states.transpose(2, 3)))
            * self.scaling_qdq(
                torch.ones(1, dtype=value_states.dtype, device=value_states.device) * self.scaling
            )
        )

        # Masked softmax
        attn_min = self.reduce_min_output_qdq(torch.amin(attn_weights, dim=-1, keepdim=True))
        attn_vv = self.minus_0_output_qdq(
            attn_min + self.neg_20_qdq(
                torch.ones(1, dtype=value_states.dtype, device=value_states.device) * (-20)
            )
        )
        attn_weights = self.where_attn_qdq(torch.where(attention_mask == 0, attn_weights, attn_vv))
        attn_weights = self.softmax_output_qdq(
            F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        )

        attn_output = self.attn_value_matmul_output_qdq(torch.matmul(attn_weights, value_states))

        # Output gating: output = output * sigmoid(gate)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        gate_sig = self.attn_gate_sigmoid_qdq(torch.sigmoid(gate))
        attn_output = self.attn_gate_mul_qdq(attn_output * gate_sig)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ============================================================================
# Decoder layers
# ============================================================================

class Qwen3_5FullAttnDecoderLayer(GradientCheckpointingLayer):
    """Full attention decoder layer with QDQ nodes."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3_5Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = QRMSNorm(config.hidden_size, eps=config.rms_norm_eps, quant_bits=16)
        self.post_attention_layernorm = QRMSNorm(config.hidden_size, eps=config.rms_norm_eps, quant_bits=16)

        # QDQ
        self.input_layernorm_input_qdq = ActivationQDQ(bits=16)
        self.add_0_lhs_input_qdq = ActivationQDQ(bits=16)
        self.add_0_output_qdq = ActivationQDQ(bits=16)
        self.add_1_lhs_input_qdq = ActivationQDQ(bits=16)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, cache_position=None,
                position_embeddings=None, **kwargs):
        hidden_states = self.input_layernorm_input_qdq(hidden_states)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.add_0_output_qdq(residual + self.add_0_lhs_input_qdq(hidden_states))
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.add_1_lhs_input_qdq(hidden_states)
        return hidden_states


class Qwen3_5GDNMlp(nn.Module):
    """Unquantized gated MLP for GDN layers, matching HF weight names."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5GDNDecoderLayer(GradientCheckpointingLayer):
    """GDN decoder layer — NO QDQ nodes (stays on CPU, unquantized)."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        # Use original HF GDN layer (no quantization)
        self.linear_attn = Qwen3_5GatedDeltaNet(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3_5GDNMlp(config)
        # Use standard GemmaRMSNorm (from HF, not quantized)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                position_ids=None, past_key_values=None, use_cache=False, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
        )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ============================================================================
# Model
# ============================================================================

class Qwen3_5PreTrainedModel(PreTrainedModel):
    config_class = HFQwen3_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5FullAttnDecoderLayer", "Qwen3_5GDNDecoderLayer"]


class Qwen3_5TextModel(nn.Module):
    def __init__(self, config: HFQwen3_5Config):
        super().__init__()
        text_config = config.text_config if hasattr(config, "text_config") else config
        self.config = text_config
        self.padding_idx = getattr(text_config, "pad_token_id", None)
        self.vocab_size = text_config.vocab_size

        self.embed_tokens = QEmbedding(text_config.vocab_size, text_config.hidden_size, self.padding_idx, quant_bits=16)

        # Build layers: GDN (unquantized) or Full Attention (quantized)
        layers = []
        for layer_idx in range(text_config.num_hidden_layers):
            if text_config.layer_types[layer_idx] == "full_attention":
                layers.append(Qwen3_5FullAttnDecoderLayer(text_config, layer_idx))
            else:
                layers.append(Qwen3_5GDNDecoderLayer(text_config, layer_idx))
        self.layers = nn.ModuleList(layers)

        self.norm = QRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps, quant_bits=16)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=text_config)
        self.gradient_checkpointing = False

        # Sin/cos cache
        self.register_buffer("mllm_max_sin_embedding", None)
        self.register_buffer("mllm_max_cos_embedding", None)
        self.sin_embedding_input_qdq = ActivationQDQ(bits=16)
        self.cos_embedding_input_qdq = ActivationQDQ(bits=16)
        self.norm_input_qdq = ActivationQDQ(bits=16)

    @torch.no_grad()
    def convert_rope_for_deploy(self):
        """Quantize RoPE tables to uint16 for deployment."""
        sin_scale = self.sin_embedding_input_qdq.fake_quant.scale
        sin_zp = self.sin_embedding_input_qdq.fake_quant.zero_point
        sin_qmin = self.sin_embedding_input_qdq.fake_quant.quant_min
        sin_qmax = self.sin_embedding_input_qdq.fake_quant.quant_max

        cos_scale = self.cos_embedding_input_qdq.fake_quant.scale
        cos_zp = self.cos_embedding_input_qdq.fake_quant.zero_point
        cos_qmin = self.cos_embedding_input_qdq.fake_quant.quant_min
        cos_qmax = self.cos_embedding_input_qdq.fake_quant.quant_max

        sin_int = torch.round(self.mllm_max_sin_embedding / sin_scale + sin_zp).clamp(sin_qmin, sin_qmax)
        self.mllm_max_sin_embedding = sin_int.to(torch.uint16)

        cos_int = torch.round(self.mllm_max_cos_embedding / cos_scale + cos_zp).clamp(cos_qmin, cos_qmax)
        self.mllm_max_cos_embedding = cos_int.to(torch.uint16)

    def _update_linear_attn_mask(self, attention_mask, past_key_values):
        linear_attn_mask = attention_mask
        has_prev = past_key_values is not None and getattr(past_key_values, "has_previous_state", False)
        if has_prev or (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None
        return linear_attn_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and not isinstance(past_key_values, Qwen3_5DynamicCache):
            past_key_values = Qwen3_5DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ) + past_seen_tokens
            # 4D position_ids: [text, temporal, height, width] — only text matters for us
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids if position_ids.ndim == 2 else None

        hidden_states = inputs_embeds

        # Causal mask for full attention layers
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        # Linear attention mask for GDN layers
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)

        # RoPE embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Compute or gather QDQ-wrapped RoPE embeddings (for QNN quantization of full attention layers)
        mllm_qualcomm_max_length = kwargs.get("mllm_qualcomm_max_length", None)
        if self.mllm_max_sin_embedding is None and self.mllm_max_cos_embedding is None:
            assert mllm_qualcomm_max_length is not None
            max_pos = torch.arange(
                0, mllm_qualcomm_max_length, device=inputs_embeds.device
            ).view(1, 1, -1).expand(3, 1, -1)
            rope_max = self.rotary_emb(hidden_states, max_pos)
            # rope_max returns (cos, sin) each [1, max_len, rotary_dim] after mrope interleaving
            self.mllm_max_cos_embedding = self.cos_embedding_input_qdq(
                rope_max[0].to(inputs_embeds.dtype)
            )
            self.mllm_max_sin_embedding = self.sin_embedding_input_qdq(
                rope_max[1].to(inputs_embeds.dtype)
            )

        # Indexed RoPE for full attention QDQ path
        if text_position_ids is not None:
            qdq_position_embeddings = (
                self.mllm_max_cos_embedding[:, text_position_ids.squeeze(0), :],
                self.mllm_max_sin_embedding[:, text_position_ids.squeeze(0), :],
            )
        else:
            qdq_position_embeddings = position_embeddings

        # Forward through layers
        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, Qwen3_5FullAttnDecoderLayer):
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=qdq_position_embeddings,
                )
            else:
                # GDN layer
                hidden_states = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=linear_attn_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

        hidden_states = self.norm(self.norm_input_qdq(hidden_states))

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen3_5Model(nn.Module):
    """Outer model wrapper matching HF checkpoint: model.language_model.layers.*"""

    def __init__(self, config: HFQwen3_5Config):
        super().__init__()
        self.language_model = Qwen3_5TextModel(config)

    def forward(self, **kwargs):
        return self.language_model(**kwargs)

    @property
    def embed_tokens(self):
        return self.language_model.embed_tokens

    def convert_rope_for_deploy(self):
        self.language_model.convert_rope_for_deploy()


class Qwen3_5ForCausalLM(Qwen3_5PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: HFQwen3_5Config):
        super().__init__(config)
        text_config = config.text_config if hasattr(config, "text_config") else config
        self.model = Qwen3_5Model(config)
        self.vocab_size = text_config.vocab_size
        self.lm_head = QLinearLPBQ(text_config.hidden_size, text_config.vocab_size, bias=False, block_size=16)
        self.lm_head_input_qdq = ActivationQDQ(bits=16)
        self.lm_head_output_qdq = ActivationQDQ(bits=16)

        self.mllm_qualcomm_max_length = 2048

        self.post_init()

    def copy_lm_head_weight_from_embed_tokens(self):
        """Copy embedding weights to lm_head for tied embeddings."""
        self.lm_head.weight = nn.Parameter(self.model.embed_tokens.weight.data.clone())

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            mllm_qualcomm_max_length=self.mllm_qualcomm_max_length,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head_output_qdq(self.lm_head(self.lm_head_input_qdq(hidden_states)))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )
