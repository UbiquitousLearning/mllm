# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from transformers.models.llama.configuration_llama import LlamaConfig

# Replace linear, rms_norm with:
from pymllm.mobile.backends.qualcomm.transformers.core.rms_norm import QRMSNorm
from pymllm.mobile.backends.qualcomm.transformers.core.qlinear import (
    QLinearLPBQ,
)
from pymllm.mobile.backends.qualcomm.transformers.core.qdq import (
    ActivationQDQ,
    FixedActivationQDQ,
)
from pymllm.mobile.backends.qualcomm.transformers.core.embedding import QEmbedding
from pymllm.mobile.backends.qualcomm.transformers.core.observer import ConcatObserver


logger = logging.get_logger(__name__)


class LlamaRMSNorm(QRMSNorm):
    def __init__(self, hidden_size, eps=1e-6, quant_bits=16):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(hidden_size, eps=eps, quant_bits=quant_bits)


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(
    x, x_observer, x2_neg_fake_quant: ActivationQDQ, concat_observer: ConcatObserver
):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return concat_observer(torch.cat((x2_neg_fake_quant(-x2), x1), dim=-1))


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = QLinearLPBQ(
            self.hidden_size,
            self.intermediate_size,
            bias=config.mlp_bias,
            block_size=32,
        )
        self.up_proj = QLinearLPBQ(
            self.hidden_size,
            self.intermediate_size,
            bias=config.mlp_bias,
            block_size=32,
        )
        self.down_proj = QLinearLPBQ(
            self.intermediate_size,
            self.hidden_size,
            bias=config.mlp_bias,
            block_size=32,
        )
        self.act_fn = ACT2FN[config.hidden_act]

        # QDQ
        self.up_proj_input_qdq = ActivationQDQ(bits=16)
        self.up_proj_output_qdq = ActivationQDQ(bits=16)
        self.gate_proj_output_qdq = ActivationQDQ(bits=16)
        self.act_output_qdq = ActivationQDQ(bits=16)
        self.down_proj_input_qdq = ActivationQDQ(bits=16)
        # For sigmoid output: scale = 1 / (q_max - q_min + 1), zp = 0
        # For 16-bit: q_min = 0, q_max = 65535
        sigmoid_scale = 1.0 / (65535 - 0 + 1)  # 1 / 65536
        self.sigmoid_output_qdq = FixedActivationQDQ(
            scale=sigmoid_scale, zero_point=0, bits=16
        )

    def forward(self, x):
        x = self.up_proj_input_qdq(x)
        up_result = self.up_proj_output_qdq(self.up_proj(x))
        gate_result = self.gate_proj_output_qdq(self.gate_proj(x))

        # SiLU or other activation
        gate_result = self.act_output_qdq(
            gate_result * self.sigmoid_output_qdq(F.sigmoid(gate_result))
        )

        o = self.down_proj_input_qdq(gate_result * up_result)
        o = self.down_proj(o)
        return o


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = QLinearLPBQ(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
            block_size=32,
        )
        self.k_proj = QLinearLPBQ(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            block_size=32,
        )
        self.v_proj = QLinearLPBQ(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            block_size=32,
        )
        self.o_proj = QLinearLPBQ(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            block_size=32,
        )

        # QDQ
        self.q_proj_input_qdq = ActivationQDQ(bits=16)
        self.k_proj_input_qdq = ActivationQDQ(bits=16)
        self.v_proj_input_qdq = ActivationQDQ(bits=16)

        self.q_proj_output_qdq = ActivationQDQ(bits=16)
        self.k_proj_output_qdq = ActivationQDQ(bits=16)

        self.q_rope_mul_0_output_qdq = ActivationQDQ(bits=16)
        self.q_rope_mul_1_output_qdq = ActivationQDQ(bits=16)
        self.q_rope_add_0_output_qdq = ActivationQDQ(bits=16)
        self.k_rope_mul_0_output_qdq = ActivationQDQ(bits=16)
        self.k_rope_mul_1_output_qdq = ActivationQDQ(bits=16)
        self.k_rope_add_0_output_qdq = ActivationQDQ(bits=16)

        self.q_rope_concat_observer = ConcatObserver(
            dtype=torch.int32,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
            quant_min=0,
            quant_max=2**16 - 1,
            eps=0.0001 / 65535,
            is_dynamic=False,
        )
        self.q_rope_neg_half_qdq = ActivationQDQ(bits=16)
        self.k_rope_concat_observer = ConcatObserver(
            dtype=torch.int32,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
            quant_min=0,
            quant_max=2**16 - 1,
            eps=0.0001 / 65535,
            is_dynamic=False,
        )
        self.k_rope_neg_half_qdq = ActivationQDQ(bits=16)
        self.k_rope_concat_observer.add_observer(
            self.k_proj_input_qdq.fake_quant.activation_post_process
        )
        self.k_rope_concat_observer.add_observer(
            self.k_rope_neg_half_qdq.fake_quant.activation_post_process
        )
        self.q_rope_concat_observer.add_observer(
            self.q_proj_input_qdq.fake_quant.activation_post_process
        )
        self.q_rope_concat_observer.add_observer(
            self.q_rope_neg_half_qdq.fake_quant.activation_post_process
        )

        # In qnn, is uint8 sym.
        self.k_cast_to_int8_qdq = ActivationQDQ(
            bits=8, qscheme=torch.per_tensor_symmetric
        )
        self.v_cast_to_int8_qdq = ActivationQDQ(
            bits=8, qscheme=torch.per_tensor_symmetric
        )

        self.v_cast_to_int16_qdq = ActivationQDQ(bits=16)
        self.qk_matmul_output_qdq = ActivationQDQ(bits=16)
        self.scaling_qdq = ActivationQDQ(bits=16)
        self.neg_20_qdq = ActivationQDQ(bits=16)
        self.reduce_min_output_qdq = ActivationQDQ(bits=16)
        self.mul_0_output_qdq = ActivationQDQ(bits=16)
        self.minus_0_output_qdq = ActivationQDQ(bits=16)
        self.softmax_output_qdq = ActivationQDQ(bits=16)
        self.attn_value_matmul_output_qdq = ActivationQDQ(bits=16)
        self.where_attn_qdq = ActivationQDQ(bits=16)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        hidden_states = self.q_proj_input_qdq(hidden_states)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states = self.q_proj_output_qdq(query_states)

        hidden_states_k = self.k_proj_input_qdq(hidden_states)
        key_states = self.k_proj(hidden_states_k).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj_output_qdq(key_states)

        hidden_states_v = self.v_proj_input_qdq(hidden_states)
        value_states = self.v_proj(hidden_states_v).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query_states = self.q_rope_add_0_output_qdq(
            self.q_rope_mul_0_output_qdq(query_states * cos)
            + self.q_rope_mul_1_output_qdq(
                rotate_half(
                    query_states,
                    self.q_proj_input_qdq.fake_quant.activation_post_process,
                    self.q_rope_neg_half_qdq,
                    self.q_rope_concat_observer,
                )
                * sin
            )
        )
        key_states = self.k_rope_add_0_output_qdq(
            self.k_rope_mul_0_output_qdq(key_states * cos)
            + self.k_rope_mul_1_output_qdq(
                rotate_half(
                    key_states,
                    self.k_proj_input_qdq.fake_quant.activation_post_process,
                    self.k_rope_neg_half_qdq,
                    self.k_rope_concat_observer,
                )
                * sin
            )
        )

        key_states = self.k_cast_to_int8_qdq(key_states)
        value_states = self.v_cast_to_int8_qdq(self.v_cast_to_int16_qdq(value_states))

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = self.mul_0_output_qdq(
            self.qk_matmul_output_qdq(
                torch.matmul(query_states, key_states.transpose(2, 3))
            )
            * self.scaling_qdq(
                torch.ones(1, dtype=value_states.dtype, device=value_states.device)
                * self.scaling
            )
        )

        attn_min = self.reduce_min_output_qdq(
            torch.amin(attn_weights, dim=-1, keepdim=True)
        )
        attn_vv = self.minus_0_output_qdq(
            attn_min
            + self.neg_20_qdq(
                torch.ones(1, dtype=value_states.dtype, device=value_states.device)
                * (-20)
            )
        )
        attn_weights = self.where_attn_qdq(
            torch.where(attention_mask == 0, attn_weights, attn_vv)
        )

        attn_weights = self.softmax_output_qdq(
            nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )
        )
        attn_output = self.attn_value_matmul_output_qdq(
            torch.matmul(attn_weights, value_states)
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, quant_bits=16
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, quant_bits=16
        )

        # QDQ
        if self.layer_idx != 0:
            self.input_layernorm_input_qdq = ActivationQDQ(bits=16)
        self.add_0_lhs_input_qdq = ActivationQDQ(bits=16)
        self.add_0_output_qdq = ActivationQDQ(bits=16)
        self.add_1_lhs_input_qdq = ActivationQDQ(bits=16)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if self.layer_idx != 0:
            hidden_states = self.input_layernorm_input_qdq(hidden_states)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.add_0_output_qdq(
            residual + self.add_0_lhs_input_qdq(hidden_states)
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.add_1_lhs_input_qdq(hidden_states)
        return hidden_states


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = QEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, quant_bits=16
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, quant_bits=16
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Register sin and cos as buffers
        self.register_buffer("mllm_max_sin_embedding", None)
        self.register_buffer("mllm_max_cos_embedding", None)
        self.sin_embedding_input_qdq = ActivationQDQ(bits=16)
        self.cos_embedding_input_qdq = ActivationQDQ(bits=16)
        self.norm_input_qdq = ActivationQDQ(bits=16)

        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def convert_rope_for_deploy(self):
        sin_scale = self.sin_embedding_input_qdq.fake_quant.scale
        sin_zero_point = self.sin_embedding_input_qdq.fake_quant.zero_point
        sin_quant_min = self.sin_embedding_input_qdq.fake_quant.quant_min
        sin_quant_max = self.sin_embedding_input_qdq.fake_quant.quant_max

        cos_scale = self.cos_embedding_input_qdq.fake_quant.scale
        cos_zero_point = self.cos_embedding_input_qdq.fake_quant.zero_point
        cos_quant_min = self.cos_embedding_input_qdq.fake_quant.quant_min
        cos_quant_max = self.cos_embedding_input_qdq.fake_quant.quant_max

        sin_int = torch.round(
            self.mllm_max_sin_embedding / sin_scale + sin_zero_point
        ).clamp(sin_quant_min, sin_quant_max)
        self.mllm_max_sin_embedding = sin_int.to(torch.uint16)

        cos_int = torch.round(
            self.mllm_max_cos_embedding / cos_scale + cos_zero_point
        ).clamp(cos_quant_min, cos_quant_max)
        self.mllm_max_cos_embedding = cos_int.to(torch.uint16)

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        if self.mllm_max_sin_embedding is None and self.mllm_max_cos_embedding is None:
            mllm_qualcomm_max_length = kwargs.get("mllm_qualcomm_max_length", None)
            assert mllm_qualcomm_max_length is not None
            max_position_ids = torch.arange(
                0,
                mllm_qualcomm_max_length,
                dtype=position_ids.dtype,
                device=position_ids.device,
            ).unsqueeze(0)
            self.mllm_max_cos_embedding, self.mllm_max_sin_embedding = self.rotary_emb(
                hidden_states, max_position_ids
            )
            self.mllm_max_cos_embedding = self.mllm_max_cos_embedding.to(
                inputs_embeds.dtype
            )
            self.mllm_max_sin_embedding = self.mllm_max_sin_embedding.to(
                inputs_embeds.dtype
            )
            self.mllm_max_cos_embedding = self.cos_embedding_input_qdq(
                self.mllm_max_cos_embedding
            )
            self.mllm_max_sin_embedding = self.sin_embedding_input_qdq(
                self.mllm_max_sin_embedding
            )

        # create position embeddings to be shared across the decoder layers
        position_embeddings = (
            self.mllm_max_cos_embedding[:, position_ids.squeeze(0), :],
            self.mllm_max_sin_embedding[:, position_ids.squeeze(0), :],
        )

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(self.norm_input_qdq(hidden_states))
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = QLinearLPBQ(
            config.hidden_size, config.vocab_size, bias=False, block_size=32
        )
        self.mllm_qualcomm_max_length = None

        self.lm_head_input_qdq = ActivationQDQ(bits=16)
        self.lm_head_output_qdq = ActivationQDQ(bits=16)

        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def copy_lm_head_weight_from_embed_tokens(self):
        if self.config.tie_word_embeddings:
            self.lm_head.weight.copy_(self.model.embed_tokens.weight)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        kwargs.update({"mllm_qualcomm_max_length": self.mllm_qualcomm_max_length})
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(
            self.lm_head_input_qdq(hidden_states[:, slice_indices, :])
        )
        logits = self.lm_head_output_qdq(logits)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaForSequenceClassification(
    GenericForSequenceClassification, LlamaPreTrainedModel
): ...


class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
    base_model_prefix = (
        "transformer"  # For BC, where `transformer` was used instead of `model`
    )


class LlamaForTokenClassification(
    GenericForTokenClassification, LlamaPreTrainedModel
): ...


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
