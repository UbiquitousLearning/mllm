from __future__ import annotations

from types import SimpleNamespace

import torch

from pymllm.models.qwen3_vl import Qwen3VLForConditionalGeneration
from pymllm.quantization.methods.compressed_tensors import CompressedTensorsConfig


def _make_vl_config() -> SimpleNamespace:
    text_config = SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        rope_scaling={"mrope_section": [2, 1, 1], "mrope_interleaved": True},
        max_position_embeddings=32,
        vocab_size=32,
    )
    return SimpleNamespace(
        text_config=text_config,
        vision_config=None,
        image_token_id=5,
        video_token_id=6,
        vision_start_token_id=4,
        tie_word_embeddings=False,
    )


def _make_w8a8_config() -> CompressedTensorsConfig:
    return CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "ignore": ["lm_head"],
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "group_size": None,
                        "strategy": "channel",
                        "symmetric": True,
                        "dynamic": False,
                        "actorder": None,
                        "type": "int",
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "strategy": "token",
                        "symmetric": True,
                        "dynamic": True,
                        "type": "int",
                    },
                }
            },
        }
    )


def _int8(shape: tuple[int, ...], value: int) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.int8)


def test_quantized_vl_text_loads_fused_qkv_and_gate_up_weight_and_scale():
    cfg = _make_vl_config()
    text_cfg = cfg.text_config
    model = Qwen3VLForConditionalGeneration(cfg, quant_config=_make_w8a8_config())

    layer0 = model.model.layers[0]
    assert layer0.self_attn.use_fused_qkv
    assert layer0.self_attn.qkv_proj is not None
    assert layer0.self_attn.q_proj is None
    assert layer0.self_attn.k_proj is None
    assert layer0.self_attn.v_proj is None
    assert layer0.mlp.use_fused_gate_up_proj
    assert layer0.mlp.gate_up_proj is not None
    assert layer0.mlp.gate_proj is None
    assert layer0.mlp.up_proj is None

    q_size = text_cfg.num_attention_heads * text_cfg.head_dim
    kv_size = text_cfg.num_key_value_heads * text_cfg.head_dim
    hidden = text_cfg.hidden_size
    inter = text_cfg.intermediate_size

    weights = {
        "model.layers.0.self_attn.q_proj.weight": _int8((q_size, hidden), 1),
        "model.layers.0.self_attn.k_proj.weight": _int8((kv_size, hidden), 2),
        "model.layers.0.self_attn.v_proj.weight": _int8((kv_size, hidden), 3),
        "model.layers.0.self_attn.q_proj.weight_scale": torch.full((q_size, 1), 0.1),
        "model.layers.0.self_attn.k_proj.weight_scale": torch.full((kv_size, 1), 0.2),
        "model.layers.0.self_attn.v_proj.weight_scale": torch.full((kv_size, 1), 0.3),
        "model.layers.0.mlp.gate_proj.weight": _int8((inter, hidden), 4),
        "model.layers.0.mlp.up_proj.weight": _int8((inter, hidden), 5),
        "model.layers.0.mlp.gate_proj.weight_scale": torch.full((inter, 1), 0.4),
        "model.layers.0.mlp.up_proj.weight_scale": torch.full((inter, 1), 0.5),
    }

    model.load_weights(weights.items())

    qkv = layer0.self_attn.qkv_proj
    assert torch.equal(qkv.weight[:q_size], weights["model.layers.0.self_attn.q_proj.weight"])
    assert torch.equal(
        qkv.weight[q_size : q_size + kv_size],
        weights["model.layers.0.self_attn.k_proj.weight"],
    )
    assert torch.equal(
        qkv.weight[q_size + kv_size : q_size + 2 * kv_size],
        weights["model.layers.0.self_attn.v_proj.weight"],
    )
    torch.testing.assert_close(
        qkv.weight_scale[:q_size],
        weights["model.layers.0.self_attn.q_proj.weight_scale"],
    )
    torch.testing.assert_close(
        qkv.weight_scale[q_size : q_size + kv_size],
        weights["model.layers.0.self_attn.k_proj.weight_scale"],
    )
    torch.testing.assert_close(
        qkv.weight_scale[q_size + kv_size : q_size + 2 * kv_size],
        weights["model.layers.0.self_attn.v_proj.weight_scale"],
    )

    gate_up = layer0.mlp.gate_up_proj
    assert torch.equal(gate_up.weight[:inter], weights["model.layers.0.mlp.gate_proj.weight"])
    assert torch.equal(gate_up.weight[inter : 2 * inter], weights["model.layers.0.mlp.up_proj.weight"])
    torch.testing.assert_close(
        gate_up.weight_scale[:inter],
        weights["model.layers.0.mlp.gate_proj.weight_scale"],
    )
    torch.testing.assert_close(
        gate_up.weight_scale[inter : 2 * inter],
        weights["model.layers.0.mlp.up_proj.weight_scale"],
    )
