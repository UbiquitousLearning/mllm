"""Inference-only Qwen3 text model for pymllm.

Implements Qwen3ForCausalLM with:
- QK-norm attention + 1D RoPE
- RadixAttention KV-cache backend
- Optional quantized Linear methods via quant_config

Adapted from pymllm's Qwen3-VL text backbone and SGLang's qwen3.py.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable, Tuple

import torch
import torch.nn as nn

from pymllm.layers import RMSNorm
from pymllm.layers.attention.radix_attention import RadixAttention
from pymllm.layers.linear import Linear, MergedLinear
from pymllm.layers.mlp import MLP
from pymllm.layers.rope import apply_rope_pos_ids

logger = logging.getLogger(__name__)


class Qwen3Attention(nn.Module):
    """Qwen3 attention with QK norm + 1D RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 32768,
        attention_bias: bool = False,
        quant_config=None,
        prefix: str = "",
    ):
        del max_position_embeddings
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.scaling = head_dim**-0.5
        self.rope_theta = rope_theta

        def _get_qm(suffix: str):
            if quant_config is None:
                return None
            return quant_config.get_quant_method(
                layer=None,
                prefix=f"{prefix}.{suffix}" if prefix else suffix,
            )

        self.use_fused_qkv = True

        if self.use_fused_qkv:
            self.qkv_proj = MergedLinear(
                hidden_size,
                [self.q_size, self.kv_size, self.kv_size],
                bias=attention_bias,
                quant_method=_get_qm("qkv_proj"),
            )
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
        else:
            self.qkv_proj = None
            self.q_proj = Linear(
                hidden_size,
                self.q_size,
                bias=attention_bias,
                quant_method=_get_qm("q_proj"),
            )
            self.k_proj = Linear(
                hidden_size,
                self.kv_size,
                bias=attention_bias,
                quant_method=_get_qm("k_proj"),
            )
            self.v_proj = Linear(
                hidden_size,
                self.kv_size,
                bias=attention_bias,
                quant_method=_get_qm("v_proj"),
            )

        self.o_proj = Linear(
            self.q_size,
            hidden_size,
            bias=attention_bias,
            quant_method=_get_qm("o_proj"),
        )

        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch,
    ) -> torch.Tensor:
        if self.use_fused_qkv:
            qkv = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))

        # Qwen3 text uses 1D RoPE with position ids from scheduler/model runner.
        if positions.ndim > 1:
            positions = positions[0]
        apply_rope_pos_ids(
            q,
            k,
            positions,
            inplace=True,
            rotary_dim=self.head_dim,
            rope_theta=self.rope_theta,
        )

        q = q.reshape(-1, self.q_size)
        k = k.reshape(-1, self.kv_size)

        attn_output = self.attn(q, k, v, forward_batch)
        return self.o_proj(attn_output)


class Qwen3DecoderLayer(nn.Module):
    """Single Qwen3 decoder layer."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        hidden_act: str,
        attention_bias: bool,
        layer_id: int,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 32768,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            max_position_embeddings=max_position_embeddings,
            attention_bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn" if prefix else "self_attn",
        )
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=hidden_act,
            use_fused_gate_up_proj=True,
            use_bias_gate_up=False,
            use_bias_down=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp" if prefix else "mlp",
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen3Model(nn.Module):
    """Qwen3 text backbone (embedding + decoder + final norm)."""

    def __init__(self, config, quant_config=None):
        super().__init__()
        tc = getattr(config, "text_config", config)

        self.hidden_size = tc.hidden_size
        self.num_hidden_layers = tc.num_hidden_layers

        self.embed_tokens = nn.Embedding(tc.vocab_size, tc.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    hidden_size=tc.hidden_size,
                    num_heads=tc.num_attention_heads,
                    num_kv_heads=tc.num_key_value_heads,
                    head_dim=getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads),
                    intermediate_size=tc.intermediate_size,
                    hidden_act=tc.hidden_act,
                    attention_bias=getattr(tc, "attention_bias", False),
                    layer_id=layer_id,
                    rope_theta=getattr(tc, "rope_theta", 1_000_000.0),
                    rms_norm_eps=tc.rms_norm_eps,
                    max_position_embeddings=getattr(tc, "max_position_embeddings", 32768),
                    quant_config=quant_config,
                    prefix=f"model.layers.{layer_id}",
                )
                for layer_id in range(tc.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        residual = None
        for layer in self.layers:
            if residual is not None and not isinstance(layer, Qwen3DecoderLayer):
                hidden_states = hidden_states + residual
                residual = None

            if isinstance(layer, Qwen3DecoderLayer):
                layer_output = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual=residual,
                )
            else:
                layer_output = layer(positions, hidden_states, forward_batch)

            if isinstance(layer_output, tuple):
                hidden_states, residual = layer_output
            else:
                hidden_states = layer_output
                residual = None

        if residual is None:
            return self.norm(hidden_states)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Inference-only Qwen3ForCausalLM."""

    def __init__(self, config, quant_config=None):
        super().__init__()
        tc = getattr(config, "text_config", config)

        self.config = tc
        self.quant_config = quant_config

        self.model = Qwen3Model(tc, quant_config=quant_config)

        tie_word_embeddings = getattr(config, "tie_word_embeddings", getattr(tc, "tie_word_embeddings", False))
        if tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = nn.Linear(tc.hidden_size, tc.vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
    ):
        _llm_t0 = time.perf_counter()
        hidden_states = self.model(input_ids, positions, forward_batch)
        _llm_ms = (time.perf_counter() - _llm_t0) * 1000.0

        if forward_batch.forward_mode.is_extend():
            forward_batch.llm_prefill_ms = _llm_ms
            forward_batch.llm_decode_ms = None
        else:
            forward_batch.llm_decode_ms = _llm_ms

        # Prefill: keep only last token logits per sequence.
        if forward_batch.forward_mode.is_extend():
            if (
                getattr(forward_batch, "extend_start_loc", None) is not None
                and getattr(forward_batch, "extend_seq_lens", None) is not None
            ):
                last_index = (
                    forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
                ).long()
                hidden_states = hidden_states[last_index]
            else:
                hidden_states = hidden_states[-1:]

        logits = torch.matmul(
            hidden_states.to(self.lm_head.weight.dtype),
            self.lm_head.weight.T,
        )

        from pymllm.executor.model_runner import LogitsProcessorOutput

        return LogitsProcessorOutput(next_token_logits=logits)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Keep compatibility with checkpoints that omit the model prefix.
            if not name.startswith("model.") and (
                name.startswith("layers.")
                or name.startswith("embed_tokens.")
                or name.startswith("norm.")
            ):
                name = f"model.{name}"

            if tie_word_embeddings and "lm_head.weight" in name:
                continue

            name = _remap_weight_name(name)

            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                loader = getattr(param, "weight_loader", None)
                if loader is not None:
                    loader(param, loaded_weight, shard_id)
                else:
                    _load_stacked_weight(param, loaded_weight, shard_id)
                handled = True
                break

            if handled:
                continue

            if name not in params_dict:
                continue

            param = params_dict[name]
            loader = getattr(param, "weight_loader", None)
            if loader is not None:
                loader(param, loaded_weight)
            elif param.data.shape == loaded_weight.shape:
                param.data.copy_(loaded_weight)
            else:
                logger.warning(
                    "Shape mismatch: param %s (%s) vs loaded (%s), skipping.",
                    name,
                    tuple(param.data.shape),
                    tuple(loaded_weight.shape),
                )


def _remap_weight_name(name: str) -> str:
    """Remap checkpoint weight names to pymllm Qwen3 parameter names."""
    if name.startswith("model.language_model."):
        name = name.replace("model.language_model.", "model.", 1)
    elif name.startswith("language_model."):
        name = name.replace("language_model.", "model.", 1)
    return name


def _load_stacked_weight(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id,
) -> None:
    """Load one shard into a fused parameter (QKV or gate_up)."""
    if isinstance(shard_id, str):
        # QKV fused layout: [Q, K, V] where Q may be wider than K/V in GQA.
        total_size = param.data.shape[0]
        shard_size = loaded_weight.shape[0]
        if shard_id == "q":
            param.data[0:shard_size].copy_(loaded_weight)
        elif shard_id == "k":
            kv_size = shard_size
            q_size = total_size - 2 * kv_size
            param.data[q_size : q_size + kv_size].copy_(loaded_weight)
        elif shard_id == "v":
            kv_size = shard_size
            q_size = total_size - 2 * kv_size
            param.data[q_size + kv_size : q_size + 2 * kv_size].copy_(loaded_weight)
    else:
        # gate_up fused layout: [gate, up]
        shard_size = loaded_weight.shape[0]
        param.data[shard_id * shard_size : (shard_id + 1) * shard_size].copy_(
            loaded_weight
        )
