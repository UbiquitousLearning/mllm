from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from pymllm.models.qwen3 import Qwen3DecoderLayer
from pymllm.models.qwen3 import Qwen3Model


class _RecordingNorm(nn.Module):
    def __init__(self, residual_offset: float):
        super().__init__()
        self.residual_offset = residual_offset
        self.seen_residual: list[bool] = []

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ):
        self.seen_residual.append(residual is not None)
        if residual is None:
            return x + 1.0
        residual_out = x + residual
        return residual_out + self.residual_offset, residual_out


class _AttentionAdd(nn.Module):
    def forward(self, positions, hidden_states, forward_batch):
        del positions, forward_batch
        return hidden_states + 3.0


class _MLPAdd(nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 4.0


class _CarryLayer(nn.Module):
    def forward(self, positions, hidden_states, forward_batch, **kwargs):
        del positions, forward_batch, kwargs
        return hidden_states + 10.0, hidden_states + 100.0


class _TensorLayer(nn.Module):
    def forward(self, positions, hidden_states, forward_batch):
        del positions, forward_batch
        return hidden_states * 2.0


def test_qwen3_decoder_layer_returns_residual_carry_and_fuses_post_attn_norm():
    layer = Qwen3DecoderLayer(
        hidden_size=2,
        num_heads=1,
        num_kv_heads=1,
        head_dim=2,
        intermediate_size=4,
        hidden_act="silu",
        attention_bias=False,
        layer_id=0,
    )
    layer.input_layernorm = _RecordingNorm(residual_offset=10.0)
    layer.post_attention_layernorm = _RecordingNorm(residual_offset=20.0)
    layer.self_attn = _AttentionAdd()
    layer.mlp = _MLPAdd()

    hidden_states = torch.tensor([[1.0, 2.0]])

    next_hidden, residual = layer(
        positions=torch.tensor([0]),
        hidden_states=hidden_states,
        forward_batch=SimpleNamespace(),
        residual=None,
    )

    assert layer.input_layernorm.seen_residual == [False]
    assert layer.post_attention_layernorm.seen_residual == [True]
    torch.testing.assert_close(residual, torch.tensor([[6.0, 8.0]]))
    torch.testing.assert_close(next_hidden, torch.tensor([[30.0, 32.0]]))


def test_qwen3_model_materializes_residual_before_tensor_returning_layer():
    cfg = SimpleNamespace(
        hidden_size=2,
        intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=32,
        attention_bias=False,
        vocab_size=8,
        hidden_act="silu",
    )
    model = Qwen3Model(cfg)
    model.layers = nn.ModuleList([_CarryLayer(), _TensorLayer()])
    model.norm = nn.Identity()

    input_embeds = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=torch.float32,
    )

    hidden_states = model(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
        positions=torch.tensor([0, 1, 2], dtype=torch.int64),
        forward_batch=SimpleNamespace(),
        input_embeds=input_embeds,
    )

    torch.testing.assert_close(
        hidden_states,
        (input_embeds + 10.0 + input_embeds + 100.0) * 2.0,
    )
