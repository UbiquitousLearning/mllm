from __future__ import annotations

import logging
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _get_text_config(config):
    """Extract text config from multimodal config, or return config as-is."""
    return getattr(config, "text_config", config)


def _get_layer_types(config) -> List[str]:
    """Return per-layer type list for Gemma 3n."""
    tc = _get_text_config(config)
    if hasattr(tc, "layer_types") and tc.layer_types is not None:
        return tc.layer_types

    n_layers = tc.num_hidden_layers
    return [
        "full_attention" if (i + 1) % 5 == 0 else "sliding_attention"
        for i in range(n_layers)
    ]


def _get_hidden_act_fn(name: str):
    if name == "silu":
        return F.silu
    if name == "gelu":
        return lambda x: F.gelu(x, approximate="none")
    if name in ("gelu_tanh", "gelu_pytorch_tanh"):
        return lambda x: F.gelu(x, approximate="tanh")
    raise ValueError(f"Unsupported hidden_act: {name}")


def _get_intermediate_size(config, layer_id: int) -> int:
    tc = _get_text_config(config)
    intermediate_size = tc.intermediate_size
    if isinstance(intermediate_size, int):
        return intermediate_size
    return int(intermediate_size[layer_id])


class SimpleGemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ):
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected last dim == hidden_size ({self.hidden_size}), "
                f"but got input shape {tuple(x.shape)}"
            )

        if residual is not None:
            residual = residual + x
            x = residual

        out = self._norm(x.float()) * self.weight.float()
        out = out.to(dtype=x.dtype)

        if residual is not None:
            return out, residual
        return out



class SimpleRMSNormNoWeight(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x_fp32 = x.float()
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        out = x_fp32 * torch.rsqrt(var + self.eps)
        return out.to(x_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _build_rope_cos_sin(
    positions: torch.Tensor,
    head_dim: int,
    base: float,
    device,
    dtype,
):
    # positions: [B, S]
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = positions[:, :, None].float() * inv_freq[None, None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, S, D], cos/sin: [B, S, D]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


class SimpleMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str,
        activation_sparsity: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_name = activation
        self.act = _get_hidden_act_fn(activation)
        self.activation_sparsity = float(activation_sparsity)

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
        target_sparsity_tensor = torch.tensor(
            self.activation_sparsity,
            dtype=torch.float32,
            device=inputs.device,
        )
        normal_dist = torch.distributions.normal.Normal(0, 1)
        std_multiplier = normal_dist.icdf(target_sparsity_tensor).to(dtype=inputs.dtype)
        inputs_mean = torch.mean(inputs, dim=-1, keepdim=True)
        inputs_std = torch.std(inputs, dim=-1, keepdim=True, unbiased=False)
        cutoff_x = inputs_mean + inputs_std * std_multiplier
        return F.relu(inputs - cutoff_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        if self.activation_sparsity > 0.0:
            gate = self._gaussian_topk(gate)
        gate = self.act(gate)
        up = self.up_proj(x)
        hidden = gate * up
        return self.down_proj(hidden)



class SimpleScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim)
        self.scalar_embed_scale = float(embed_scale)
        self.register_buffer("embed_scale", torch.tensor(float(embed_scale)), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class SimpleLaurelBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        tc = _get_text_config(config)
        laurel_rank = getattr(tc, "laurel_rank", 8)
        self.linear_left = nn.Linear(tc.hidden_size, laurel_rank, bias=False)
        self.linear_right = nn.Linear(laurel_rank, tc.hidden_size, bias=False)
        self.post_laurel_norm = SimpleGemmaRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        laurel_hidden_states = self.linear_left(hidden_states)
        laurel_hidden_states = self.linear_right(laurel_hidden_states)
        normed = self.post_laurel_norm(laurel_hidden_states)
        return hidden_states + normed


class SimpleAltUp(nn.Module):
    def __init__(self, config):
        super().__init__()
        tc = _get_text_config(config)
        self.config = tc
        self.altup_num_inputs = getattr(tc, "altup_num_inputs", 2)
        self.altup_active_idx = getattr(tc, "altup_active_idx", 0)
        self.correct_output_scale = nn.Parameter(torch.zeros(tc.hidden_size))
        self.correction_coefs = nn.Linear(self.altup_num_inputs, self.altup_num_inputs, bias=False)
        self.prediction_coefs = nn.Linear(self.altup_num_inputs, self.altup_num_inputs ** 2, bias=False)
        self.modality_router = nn.Linear(tc.hidden_size, self.altup_num_inputs, bias=False)
        self.router_norm = SimpleGemmaRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)
        self.register_buffer("router_input_scale", torch.tensor(tc.hidden_size ** -1.0), persistent=False)

    def compute_router_modalities(self, x: torch.Tensor) -> torch.Tensor:
        router_inputs = self.router_norm(x) * self.router_input_scale.to(dtype=x.dtype, device=x.device)
        routed = self.modality_router(router_inputs)
        return torch.tanh(routed.float()).to(dtype=x.dtype, device=x.device)

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [P, B, S, H]
        modalities = self.compute_router_modalities(hidden_states[self.altup_active_idx])
        all_coefs = self.prediction_coefs(modalities).reshape(
            *modalities.shape[:-1], self.altup_num_inputs, self.altup_num_inputs
        )
        all_coefs = all_coefs.permute(0, 1, 3, 2)  # [B, S, P, P]
        predictions = torch.matmul(hidden_states.permute(1, 2, 3, 0), all_coefs)
        predictions = predictions.permute(3, 0, 1, 2).contiguous()
        predictions = predictions + hidden_states
        return predictions.to(dtype=hidden_states.dtype)

    def correct(self, predictions: torch.Tensor, activated: torch.Tensor) -> torch.Tensor:
        # predictions: [P, B, S, H], activated: [B, S, H]
        modalities = self.compute_router_modalities(activated)
        innovation = activated - predictions[self.altup_active_idx]
        innovation = innovation.repeat(self.altup_num_inputs, 1, 1, 1)

        all_coefs = self.correction_coefs(modalities) + 1.0
        all_coefs = all_coefs.permute(2, 0, 1).unsqueeze(-1)  # [P, B, S, 1]

        corrected = innovation * all_coefs
        corrected = corrected + predictions
        return corrected.to(dtype=activated.dtype)

    def forward(self, corrected: torch.Tensor) -> torch.Tensor:
        return (corrected.to(dtype=self.correct_output_scale.dtype) * self.correct_output_scale).to(dtype=corrected.dtype)

    def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor:
        return self.forward(corrected)


class Gemma3nAttention(nn.Module):
    """Text-only Gemma 3n attention v1.

    Implements the text attention path used by the native Gemma3n
    development mode, including layer-specific RoPE parameters and
    full/sliding causal attention masks.
    """

    def __init__(self, config, layer_id: int):
        super().__init__()
        tc = _get_text_config(config)
        self.layer_id = layer_id
        self.layer_type = _get_layer_types(config)[layer_id]

        self.hidden_size = tc.hidden_size
        self.num_heads = tc.num_attention_heads
        self.num_kv_heads = tc.num_key_value_heads
        self.head_dim = getattr(tc, "head_dim", self.hidden_size // self.num_heads)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, self.hidden_size, bias=False)

        self.q_norm = SimpleGemmaRMSNorm(self.head_dim)
        self.k_norm = SimpleGemmaRMSNorm(self.head_dim)
        self.v_norm = SimpleRMSNormNoWeight(eps=getattr(tc, "rms_norm_eps", 1e-6))

        self.sliding_window = getattr(tc, "sliding_window", None)

        num_kv_shared_layers = int(getattr(tc, "num_kv_shared_layers", 0))
        first_kv_shared_layer_idx = int(tc.num_hidden_layers) - num_kv_shared_layers
        layer_types = _get_layer_types(config)
        prev_layers = layer_types[:first_kv_shared_layer_idx]

        self.is_kv_shared_layer = layer_id >= first_kv_shared_layer_idx > 0
        if self.is_kv_shared_layer:
            self.kv_shared_layer_index = (
                len(prev_layers) - 1 - prev_layers[::-1].index(self.layer_type)
            )
            self.store_full_length_kv = False
        else:
            self.kv_shared_layer_index = None
            if prev_layers and self.layer_type in prev_layers:
                self.store_full_length_kv = (
                    layer_id == len(prev_layers) - 1 - prev_layers[::-1].index(self.layer_type)
                )
            else:
                self.store_full_length_kv = False

        rope_parameters = getattr(tc, "rope_parameters", None)
        rope_theta = None
        if isinstance(rope_parameters, dict):
            layer_rope = rope_parameters.get(self.layer_type, {})
            if isinstance(layer_rope, dict):
                rope_theta = layer_rope.get("rope_theta", None)

        if rope_theta is None:
            if self.layer_type == "sliding_attention":
                rope_theta = getattr(tc, "rope_local_base_freq", 10000.0)
            else:
                rope_theta = getattr(tc, "rope_theta", 10000.0)

        self.rope_theta = float(rope_theta)

    def _build_attention_mask(
        self,
        positions: torch.Tensor,
        device,
    ) -> torch.Tensor:
        # positions: [B, S]
        q_pos = positions[:, :, None]   # [B, S, 1]
        k_pos = positions[:, None, :]   # [B, 1, S]

        # causal
        mask = k_pos <= q_pos

        # local sliding window for sliding_attention layers
        if self.layer_type == "sliding_attention" and self.sliding_window is not None:
            lower_bound = q_pos - (int(self.sliding_window) - 1)
            mask = mask & (k_pos >= lower_bound)

        return mask

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: Any,
    ) -> torch.Tensor:
        positions = positions.to(device=hidden_states.device, non_blocking=True)
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # [B, H, S, D]

        cos, sin = _build_rope_cos_sin(
            positions=positions,
            head_dim=self.head_dim,
            base=self.rope_theta,
            device=hidden_states.device,
            dtype=q.dtype,
        )
        q = _apply_rope(q, cos, sin)

        shared_kv_cache = None
        if isinstance(forward_batch, dict):
            shared_kv_cache = forward_batch.get("kv_shared_cache")
        elif forward_batch is not None and hasattr(forward_batch, "kv_shared_cache"):
            shared_kv_cache = getattr(forward_batch, "kv_shared_cache")

        if (
            self.is_kv_shared_layer
            and shared_kv_cache is not None
            and self.kv_shared_layer_index in shared_kv_cache
        ):
            k, v = shared_kv_cache[self.kv_shared_layer_index]
            k = k.to(device=q.device, dtype=q.dtype, non_blocking=True)
            v = v.to(device=q.device, dtype=q.dtype, non_blocking=True)
        else:
            k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            k = self.k_norm(k).transpose(1, 2)  # [B, KV, S, D]
            k = _apply_rope(k, cos, sin)

            v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            v = self.v_norm(v).transpose(1, 2)  # [B, KV, S, D]

            if shared_kv_cache is not None and self.store_full_length_kv:
                shared_kv_cache[self.layer_id] = (k, v)

        if self.num_heads != self.num_kv_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k[:, :, None, :, :].expand(batch_size, self.num_kv_heads, n_rep, seq_len, self.head_dim)
            v = v[:, :, None, :, :].expand(batch_size, self.num_kv_heads, n_rep, seq_len, self.head_dim)
            k = k.reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            v = v.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        attn_mask = self._build_attention_mask(positions, hidden_states.device)  # [B, S, S]
        mask_value = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(~attn_mask[:, None, :, :], mask_value)

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(attn_scores.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.q_size)

        return self.o_proj(attn_output)


class Gemma3nDecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        tc = _get_text_config(config)

        self.self_attn = Gemma3nAttention(config, layer_id)
        activation_sparsity_pattern = getattr(tc, "activation_sparsity_pattern", None)
        if isinstance(activation_sparsity_pattern, (list, tuple)):
            layer_activation_sparsity = float(activation_sparsity_pattern[layer_id])
        else:
            layer_activation_sparsity = float(getattr(tc, "activation_sparsity", 0.0))

        self.mlp = SimpleMLP(
            hidden_size=tc.hidden_size,
            intermediate_size=_get_intermediate_size(config, layer_id),
            activation=getattr(tc, "hidden_activation", getattr(tc, "hidden_act", "silu")),
            activation_sparsity=layer_activation_sparsity,
        )
        self.input_layernorm = SimpleGemmaRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)
        self.post_attention_layernorm = SimpleGemmaRMSNorm(
            tc.hidden_size, eps=tc.rms_norm_eps
        )
        self.pre_feedforward_layernorm = SimpleGemmaRMSNorm(
            tc.hidden_size, eps=tc.rms_norm_eps
        )
        self.post_feedforward_layernorm = SimpleGemmaRMSNorm(
            tc.hidden_size, eps=tc.rms_norm_eps
        )

        self.altup = SimpleAltUp(config)
        self.laurel = SimpleLaurelBlock(config)
        self.hidden_size_per_layer_input = getattr(tc, "hidden_size_per_layer_input", 8)
        self.per_layer_input_gate = nn.Linear(
            tc.hidden_size, self.hidden_size_per_layer_input, bias=False
        )
        self.per_layer_projection = nn.Linear(
            self.hidden_size_per_layer_input, tc.hidden_size, bias=False
        )
        self.post_per_layer_input_norm = SimpleGemmaRMSNorm(
            tc.hidden_size, eps=tc.rms_norm_eps
        )
        self.per_layer_input_act = _get_hidden_act_fn(
            getattr(tc, "hidden_activation", getattr(tc, "hidden_act", "silu"))
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,   # [P, B, S, H]
        residual: Optional[torch.Tensor],
        forward_batch: Any,
        per_layer_input: Optional[torch.Tensor] = None,  # [B, S, Pdim]
    ):
        del residual
        predictions = self.altup.predict(hidden_states)
        active_prediction = predictions[self.altup.altup_active_idx]  # [B, S, H]

        active_prediction_normed = self.input_layernorm(active_prediction)
        laurel_output = self.laurel(active_prediction_normed)

        attn_output = self.self_attn(positions, active_prediction_normed, forward_batch)
        attn_output = self.post_attention_layernorm(attn_output)

        attn_gated = active_prediction + attn_output
        attn_laurel = (attn_gated + laurel_output) / (2.0 ** 0.5)

        attn_norm = self.pre_feedforward_layernorm(attn_laurel)
        attn_ffw = self.mlp(attn_norm)
        attn_ffw_norm = self.post_feedforward_layernorm(attn_ffw)
        attn_ffw_laurel_gated = attn_laurel + attn_ffw_norm

        corrected_predictions = self.altup.correct(predictions, attn_ffw_laurel_gated)

        first_prediction = corrected_predictions[self.altup.altup_active_idx].clone()
        if getattr(self.altup.config, "altup_correct_scale", False):
            first_prediction = self.altup.scale_corrected_output(first_prediction)

        if per_layer_input is not None:
            first_prediction = self.per_layer_input_gate(first_prediction)
            first_prediction = self.per_layer_input_act(first_prediction)
            first_prediction = first_prediction * per_layer_input
            first_prediction = self.per_layer_projection(first_prediction)
            first_prediction = self.post_per_layer_input_norm(first_prediction)
            if corrected_predictions.shape[0] > 1:
                corrected_predictions[1:] = corrected_predictions[1:] + first_prediction.unsqueeze(0)

        return corrected_predictions


class Gemma3nModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        tc = _get_text_config(config)

        self.config = config
        self.embed_tokens = SimpleScaledWordEmbedding(tc.vocab_size, tc.hidden_size, embed_scale=tc.hidden_size ** 0.5)
        self.layers = nn.ModuleList(
            [Gemma3nDecoderLayer(config, i) for i in range(tc.num_hidden_layers)]
        )
        self.norm = SimpleGemmaRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)

        # Text-only modules whose names match the official Gemma3n checkpoint.
        self.hidden_size_per_layer_input = getattr(tc, "hidden_size_per_layer_input", 8)
        self.vocab_size_per_layer_input = getattr(tc, "vocab_size_per_layer_input", tc.vocab_size)
        self.altup_num_inputs = getattr(tc, "altup_num_inputs", 2)

        self.embed_tokens_per_layer = SimpleScaledWordEmbedding(
            self.vocab_size_per_layer_input,
            tc.num_hidden_layers * self.hidden_size_per_layer_input,
            embed_scale=self.hidden_size_per_layer_input ** 0.5,
        )
        self.per_layer_model_projection = nn.Linear(
            tc.hidden_size,
            tc.num_hidden_layers * self.hidden_size_per_layer_input,
            bias=False,
        )
        self.per_layer_projection_norm = SimpleGemmaRMSNorm(
            self.hidden_size_per_layer_input, eps=tc.rms_norm_eps
        )

        self.altup_projections = nn.ModuleList(
            [nn.Linear(tc.hidden_size, tc.hidden_size, bias=False) for _ in range(1, self.altup_num_inputs)]
        )
        self.altup_unembed_projections = nn.ModuleList(
            [nn.Linear(tc.hidden_size, tc.hidden_size, bias=False) for _ in range(1, self.altup_num_inputs)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any = None,
    ):
        model_device = self.norm.weight.device
        # Native Gemma3nModel accepts either flat mllm tensors [T]
        # or batched tensors [B, T]. Normalize them once here.
        if input_ids.dim() == 1:
            input_ids_hf = input_ids.unsqueeze(0)
        else:
            input_ids_hf = input_ids

        if positions is None:
            positions = torch.arange(
                input_ids_hf.shape[1],
                dtype=torch.long,
                device=input_ids_hf.device,
            ).unsqueeze(0)
        elif positions.dim() == 1:
            positions = positions.unsqueeze(0)

        # Server/scheduler tensors may arrive on CUDA even when Gemma3n
        # is intentionally instantiated on CPU for memory reasons.
        # Keep position tensors on the same device as the native model.
        positions = positions.to(device=model_device, non_blocking=True)

        embed_input_ids = input_ids_hf.to(
            device=self.embed_tokens.weight.device,
            non_blocking=True,
        )
        hidden_states_0 = self.embed_tokens(embed_input_ids).to(
            device=model_device,
            non_blocking=True,
        )
        batch_size, seq_len, _ = hidden_states_0.shape
        num_layers = len(self.layers)

        per_layer_input_ids = input_ids_hf.to(
            device=self.embed_tokens_per_layer.weight.device,
            non_blocking=True,
        )
        per_layer_inputs = self.embed_tokens_per_layer(per_layer_input_ids)
        per_layer_inputs = per_layer_inputs.to(
            device=hidden_states_0.device,
            dtype=hidden_states_0.dtype,
            non_blocking=True,
        ).reshape(
            batch_size,
            seq_len,
            num_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_model_projection(hidden_states_0)
        per_layer_projection = per_layer_projection * (hidden_states_0.shape[-1] ** -0.5)
        per_layer_projection = per_layer_projection.reshape(
            batch_size,
            seq_len,
            num_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        per_layer_inputs = (per_layer_projection + per_layer_inputs) * (2.0 ** -0.5)

        target_magnitude = torch.mean(hidden_states_0 ** 2, dim=-1, keepdim=True).clamp_min(1e-5).sqrt()

        temp_hidden_states = [hidden_states_0]
        for i in range(1, self.altup_num_inputs):
            altup_proj = self.altup_projections[i - 1](hidden_states_0)
            current_hidden_state = altup_proj.to(dtype=hidden_states_0.dtype, device=hidden_states_0.device)
            new_magnitude = torch.mean(current_hidden_state ** 2, dim=-1, keepdim=True).clamp_min(1e-5).sqrt()
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)

        hidden_states = torch.stack(temp_hidden_states, dim=0)  # [P, B, S, H]

        native_forward_batch = {"kv_shared_cache": {}}

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=None,
                forward_batch=native_forward_batch,
                per_layer_input=per_layer_inputs[:, :, layer_idx, :],
            )

        target_magnitude = torch.mean(hidden_states[0] ** 2, dim=-1, keepdim=True).clamp_min(1e-5).sqrt()
        temp_hidden_states = [hidden_states[0]]
        for i in range(1, self.altup_num_inputs):
            altup_unemb = self.altup_unembed_projections[i - 1](hidden_states[i])
            current_hidden_state = altup_unemb.to(dtype=hidden_states_0.dtype, device=hidden_states_0.device)
            new_magnitude = torch.mean(current_hidden_state ** 2, dim=-1, keepdim=True).clamp_min(1e-5).sqrt()
            current_hidden_state = current_hidden_state * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)

        hidden_states = torch.stack(temp_hidden_states, dim=0)
        hidden_states = torch.mean(hidden_states, dim=0)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Gemma3nForCausalLM(nn.Module):
    """Text-only Gemma3n Causal LM wrapper.

    The default server path uses the official HF text model and keeps the main
    token embedding and lm_head on CUDA, while offloading the large per-layer
    embedding table to CPU during weight loading. Native mode can be enabled
    for development with MLLM_GEMMA3N_NATIVE=1.
    """

    requires_cpu_first_weight_loading = False

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        text_config = _get_text_config(config)
        text_config._attn_implementation = "eager"

        import os
        self.use_native = os.environ.get("MLLM_GEMMA3N_NATIVE", "0") == "1"
        self.use_model_path_weight_loader = self.use_native
        if self.use_native:
            self.config = text_config
            self.quant_config = quant_config
            self.prefix = prefix
            self.model = Gemma3nModel(config)
            self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
            self._hf_past_key_values = None
            # Minimal batch=1 decode cache for pymllm-server native smoke tests.
            # During prefill the server passes the full prompt; during decode it
            # passes only the latest token. Until native attention is integrated
            # with mllm's KV cache, keep the full token/position history here and
            # recompute the full context, returning only the last-token logits.
            self._native_cached_input_ids = None
            self._native_cached_positions = None
            return

        from transformers.models.gemma3n.modeling_gemma3n import (
            Gemma3nForCausalLM as HFGemma3nForCausalLM,
        )

        self.config = text_config
        self.quant_config = quant_config
        self.prefix = prefix

        hf_model = HFGemma3nForCausalLM(text_config)
        self.model = hf_model.model
        self.lm_head = hf_model.lm_head

        # Minimal HF cache for current text-only server path.
        # mllm supplies only the newest token during decode, so the HF wrapper
        # must retain and reuse past_key_values across decode steps.
        self._hf_past_key_values = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any = None,
    ):
        # mllm passes flat tensors for extend/prefill, e.g. input_ids=[T] and positions=[T].
        # HF Gemma3n expects batched text tensors:
        #   input_ids: [B, T]
        #   position_ids: [B, T]
        #   inputs_embeds: [B, T, H]
        # Current server path only supports batch_size=1 for Gemma3n text-only.
        del forward_batch

        if getattr(self, "use_native", False):
            if input_ids.dim() == 1:
                input_ids_hf = input_ids.unsqueeze(0)
            else:
                input_ids_hf = input_ids

            if positions.dim() == 1:
                position_ids_hf = positions.unsqueeze(0)
            else:
                position_ids_hf = positions

            # Native text-only path currently keeps large Gemma3n weights on CPU.
            # Keep the recomputed full-context forward on CPU as well; otherwise
            # server decode can pass CUDA token tensors while embeddings/lm_head
            # remain on CPU, producing a different path from the verified local
            # stepwise generation.
            # Important: do a blocking CPU copy here. In server decode, the
            # CUDA input tensor can be reused/cleared by the runtime after the
            # forward call is scheduled; a non_blocking CUDA->CPU copy may then
            # observe zeros or stale values.
            input_ids_hf = input_ids_hf.detach().to(device=torch.device("cpu"), non_blocking=False).clone()
            position_ids_hf = position_ids_hf.detach().to(device=torch.device("cpu"), non_blocking=False).clone()

            # pymllm-server prefill sends the full prompt, while decode sends
            # only the latest token. For the current text-only native path,
            # recompute from the full cached sequence so multi-token server
            # decode matches direct greedy generation.
            is_prefill = (
                input_ids_hf.shape[1] > 1
                or self._native_cached_input_ids is None
                or self._native_cached_positions is None
            )
            if is_prefill:
                full_input_ids = input_ids_hf
            else:
                full_input_ids = torch.cat(
                    [
                        self._native_cached_input_ids.to(device=input_ids_hf.device),
                        input_ids_hf,
                    ],
                    dim=1,
                )

            # Recompute contiguous full-context positions, matching direct greedy
            # generation. Decode-time positions from the server correspond to
            # the submitted token batch, not necessarily to the recomputed full
            # native context.
            full_positions = torch.arange(
                full_input_ids.shape[1],
                dtype=torch.long,
                device=full_input_ids.device,
            ).unsqueeze(0).expand(full_input_ids.shape[0], -1)

            self._native_cached_input_ids = full_input_ids.detach().cpu()
            self._native_cached_positions = full_positions.detach().cpu()

            hidden_states = self.model(input_ids=full_input_ids, positions=full_positions)
            output_device = hidden_states.device
            logits = self.lm_head(hidden_states.to(device=self.lm_head.weight.device))

            final_logit_softcapping = getattr(self.config, "final_logit_softcapping", None)
            if final_logit_softcapping is not None:
                logits = logits / final_logit_softcapping
                logits = torch.tanh(logits)
                logits = logits * final_logit_softcapping

            # For decode, the server expects logits for the token(s) just
            # submitted in this forward call, not the whole recomputed prefix.
            logits = logits[:, -input_ids_hf.shape[1]:, :]

            return logits.to(device=output_device, non_blocking=True)

        if input_ids.dim() == 1:
            input_ids_hf = input_ids.unsqueeze(0)
        else:
            input_ids_hf = input_ids

        if positions.dim() == 1:
            position_ids_hf = positions.unsqueeze(0)
        else:
            position_ids_hf = positions

        # For batch_size=1 text-only serving:
        # - prefill has sequence length > 1, so reset HF cache;
        # - decode has sequence length == 1, so reuse stored HF cache.
        is_prefill = input_ids_hf.shape[1] > 1 or self._hf_past_key_values is None
        past_key_values = None if is_prefill else self._hf_past_key_values

        model_device = self.model.norm.weight.device
        embed_dev = self.model.embed_tokens.weight.device
        per_layer_dev = self.model.embed_tokens_per_layer.weight.device

        # If both embedding tables already live on the same device as the HF text model,
        # use the native HF text-model path directly. This is the closest apples-to-apples
        # comparison with the official implementation.
        if embed_dev == model_device and per_layer_dev == model_device:
            outputs = self.model(
                input_ids=input_ids_hf.to(device=model_device, non_blocking=True),
                per_layer_inputs=None,
                attention_mask=None,
                position_ids=position_ids_hf.to(device=model_device, non_blocking=True),
                past_key_values=past_key_values,
                inputs_embeds=None,
                use_cache=True,
            )
        else:
            embed_input_ids = input_ids_hf.to(
                device=embed_dev,
                non_blocking=True,
            )
            inputs_embeds = self.model.embed_tokens(embed_input_ids).to(
                device=model_device,
                non_blocking=True,
            )

            per_layer_input_ids = input_ids_hf.to(
                device=per_layer_dev,
                non_blocking=True,
            )
            per_layer_inputs = self.model.get_per_layer_inputs(per_layer_input_ids).to(
                device=model_device,
                dtype=inputs_embeds.dtype,
                non_blocking=True,
            )

            outputs = self.model(
                input_ids=None,
                per_layer_inputs=per_layer_inputs,
                attention_mask=None,
                position_ids=position_ids_hf.to(device=model_device, non_blocking=True),
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
            )

        if hasattr(outputs, "past_key_values"):
            self._hf_past_key_values = outputs.past_key_values

        hidden_states = outputs.last_hidden_state
        output_device = hidden_states.device
        hidden_states_for_head = hidden_states.to(
            device=self.lm_head.weight.device,
            non_blocking=True,
        )
        logits = self.lm_head(hidden_states_for_head)

        final_logit_softcapping = getattr(self.config, "final_logit_softcapping", None)
        if final_logit_softcapping is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping

        return logits.to(device=output_device, non_blocking=True)


    def load_weights_from_model_path(self, model_path, chunk_bytes: int = 256 * 1024 * 1024):
        """Load Gemma3n text-only weights directly from a local HF checkpoint.

        This path avoids materializing full safetensors shards or the full
        iterator in memory. It is needed because Gemma3n has very large text
        embedding tables, especially embed_tokens_per_layer.weight.
        """
        import gc
        import json
        from pathlib import Path

        from safetensors import safe_open

        model_path = Path(model_path)
        own_state = self.state_dict()

        def _normalize_name(name: str):
            if (
                name.startswith("model.audio_tower.")
                or name.startswith("model.vision_tower.")
                or name.startswith("model.embed_audio.")
                or name.startswith("model.embed_vision.")
            ):
                return None, "ignored_multimodal"

            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model."):]
            elif name.startswith("language_model."):
                name = "model." + name[len("language_model."):]

            return name, None

        def _tensor_nbytes_from_shape(shape, element_size: int):
            numel = 1
            for dim in shape:
                numel *= int(dim)
            return numel * element_size

        def _copy_tensor_streaming(safe_file, raw_name: str, target: torch.Tensor):
            tensor_slice = safe_file.get_slice(raw_name)
            shape = tuple(tensor_slice.get_shape())

            if tuple(target.shape) != shape:
                raise RuntimeError(
                    f"shape mismatch for {raw_name}: "
                    f"target={tuple(target.shape)} checkpoint={shape}"
                )

            estimated_bytes = _tensor_nbytes_from_shape(shape, target.element_size())

            if len(shape) >= 2 and estimated_bytes >= chunk_bytes:
                row_numel = 1
                for dim in shape[1:]:
                    row_numel *= int(dim)

                rows_per_chunk = max(1, chunk_bytes // (row_numel * target.element_size()))
                total_rows = int(shape[0])

                logger.info(
                    "Gemma3n streaming large tensor %s shape=%s rows_per_chunk=%d",
                    raw_name,
                    shape,
                    rows_per_chunk,
                )

                with torch.no_grad():
                    for start in range(0, total_rows, rows_per_chunk):
                        end = min(total_rows, start + rows_per_chunk)
                        piece = tensor_slice[start:end]
                        target[start:end].copy_(
                            piece.to(dtype=target.dtype, device=target.device)
                        )
                        del piece
                        gc.collect()
            else:
                piece = safe_file.get_tensor(raw_name)
                with torch.no_grad():
                    target.copy_(piece.to(dtype=target.dtype, device=target.device))
                del piece

        index_path = model_path / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            weight_map = index.get("weight_map", {})
            by_file = {}
            for raw_name, filename in weight_map.items():
                by_file.setdefault(filename, []).append(raw_name)
        else:
            st_files = sorted(model_path.glob("*.safetensors"))
            if not st_files:
                logger.info(
                    "Gemma3n streaming loader found no safetensors under %s; "
                    "falling back to regular load_weights path.",
                    model_path,
                )
                return self.load_weights([])

            by_file = {}
            for path in st_files:
                with safe_open(path, framework="pt", device="cpu") as safe_file:
                    by_file[path.name] = list(safe_file.keys())

        loaded = []
        skipped = []
        normalized_weight_names = set()

        logger.info(
            "Gemma3n streaming load begin. model_path=%s shards=%d",
            model_path,
            len(by_file),
        )

        for filename in sorted(by_file.keys()):
            shard_path = model_path / filename
            raw_names = by_file[filename]
            logger.info(
                "Gemma3n streaming shard begin: %s tensors=%d",
                filename,
                len(raw_names),
            )

            with safe_open(shard_path, framework="pt", device="cpu") as safe_file:
                for raw_name in raw_names:
                    name, skip_reason = _normalize_name(raw_name)

                    if name is None:
                        skipped.append((raw_name, skip_reason))
                        continue

                    normalized_weight_names.add(name)

                    if name not in own_state:
                        skipped.append((raw_name, f"missing_in_model -> {name}"))
                        continue

                    target = own_state[name]
                    shape = tuple(safe_file.get_slice(raw_name).get_shape())

                    if tuple(target.shape) != shape:
                        skipped.append(
                            (
                                raw_name,
                                f"shape_mismatch mapped={name} "
                                f"model={tuple(target.shape)} ckpt={shape}",
                            )
                        )
                        continue

                    _copy_tensor_streaming(safe_file, raw_name, target)
                    loaded.append((raw_name, name))

            logger.info(
                "Gemma3n streaming shard end: %s loaded_total=%d skipped_total=%d",
                filename,
                len(loaded),
                len(skipped),
            )
            gc.collect()

        if "lm_head.weight" in own_state and "model.embed_tokens.weight" in own_state:
            with torch.no_grad():
                own_state["lm_head.weight"].copy_(own_state["model.embed_tokens.weight"])
            loaded.append(("always_tied_from_embed_tokens", "lm_head.weight"))
            normalized_weight_names.add("lm_head.weight")

        missing_in_ckpt = [
            name for name in own_state.keys()
            if name not in normalized_weight_names
        ]

        logger.info(
            "Gemma3n streaming load end: loaded=%d skipped=%d missing_in_ckpt=%d",
            len(loaded),
            len(skipped),
            len(missing_in_ckpt),
        )

        if skipped:
            logger.info("Gemma3n streaming load first_skipped=%s", skipped[:20])

        return {
            "loaded": loaded,
            "skipped": skipped,
            "missing_in_ckpt": missing_in_ckpt,
        }

    def load_weights(self, weights):
        if hasattr(weights, "state_dict"):
            weights = weights.state_dict()

        if isinstance(weights, dict):
            weight_items = list(weights.items())
        else:
            try:
                weight_items = list(weights)
            except TypeError:
                raise TypeError(
                    f"weights must be a dict-like state_dict, a module with state_dict(), "
                    f"or an iterable of (name, tensor), got {type(weights)}"
                )

        def _normalize_name(name: str):
            # Ignore clearly multimodal-only weights for current text-only path.
            if (
                name.startswith("model.audio_tower.")
                or name.startswith("model.vision_tower.")
                or name.startswith("model.embed_audio.")
                or name.startswith("model.embed_vision.")
            ):
                return None, "ignored_multimodal"

            # HF Gemma3n outer model wraps text weights under model.language_model.*
            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model."):]
            elif name.startswith("language_model."):
                name = "model." + name[len("language_model."):]

            return name, None

        # Keep the main token embedding path on GPU for output quality.
        # Only offload the very large per-layer embedding table to CPU.
        self.model.embed_tokens_per_layer = self.model.embed_tokens_per_layer.to("cpu")

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Gemma3n devices: embed_tokens=%s embed_tokens_per_layer=%s lm_head=%s", self.model.embed_tokens.weight.device, self.model.embed_tokens_per_layer.weight.device, self.lm_head.weight.device)

        own_state = self.state_dict()
        loaded = []
        skipped = []
        normalized_weight_names = set()

        for raw_name, tensor in weight_items:
            name, skip_reason = _normalize_name(raw_name)

            if name is None:
                skipped.append((raw_name, skip_reason))
                continue

            normalized_weight_names.add(name)

            if name not in own_state:
                skipped.append((raw_name, f"missing_in_model -> {name}"))
                continue

            if own_state[name].shape != tensor.shape:
                skipped.append(
                    (
                        raw_name,
                        f"shape_mismatch mapped={name} "
                        f"model={tuple(own_state[name].shape)} ckpt={tuple(tensor.shape)}",
                    )
                )
                continue

            own_state[name].copy_(
                tensor.to(dtype=own_state[name].dtype, device=own_state[name].device)
            )
            loaded.append((raw_name, name))

        # Always tie lm_head to embed_tokens for text-only Gemma3n,
        # matching the official tied-weights behavior.
        if "lm_head.weight" in own_state and "model.embed_tokens.weight" in own_state:
            own_state["lm_head.weight"].copy_(own_state["model.embed_tokens.weight"])
            loaded.append(("always_tied_from_embed_tokens", "lm_head.weight"))
            normalized_weight_names.add("lm_head.weight")

        missing_in_ckpt = [name for name in own_state.keys() if name not in normalized_weight_names]

        logger.info(
            "Gemma3nForCausalLM.load_weights: loaded=%d skipped=%d missing_in_ckpt=%d",
            len(loaded),
            len(skipped),
            len(missing_in_ckpt),
        )

        if loaded:
            logger.info("Gemma3nForCausalLM.load_weights: first_loaded=%s", loaded[:20])
        if skipped:
            logger.info("Gemma3nForCausalLM.load_weights: first_skipped=%s", skipped[:20])

        return {
            "loaded": loaded,
            "skipped": skipped,
            "missing_in_ckpt": missing_in_ckpt,
        }


class Gemma3nForConditionalGeneration(Gemma3nForCausalLM):
    """Text-only compatibility entry point for Gemma3n checkpoints.

    The official checkpoint architecture resolves to this class, while the
    current implementation supports the language-model path.
    """
    pass
