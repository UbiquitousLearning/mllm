from __future__ import annotations

from types import SimpleNamespace

import torch

from pymllm.models.qwen3 import Qwen3ForCausalLM
from pymllm.quantization.methods.compressed_tensors import CompressedTensorsConfig


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        attention_bias=False,
        vocab_size=32,
        tie_word_embeddings=False,
        hidden_act="silu",
    )


def _make_weight(shape: tuple[int, ...], start: int) -> torch.Tensor:
    numel = 1
    for s in shape:
        numel *= s
    return torch.arange(start, start + numel, dtype=torch.float32).reshape(shape)


def _make_int8_weight(shape: tuple[int, ...], value: int) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.int8)


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


def _build_language_weights(cfg: SimpleNamespace, layer_prefix: str = "model"):
    q_size = cfg.num_attention_heads * cfg.head_dim
    kv_size = cfg.num_key_value_heads * cfg.head_dim
    hidden = cfg.hidden_size
    inter = cfg.intermediate_size

    weights = {
        f"{layer_prefix}.embed_tokens.weight": _make_weight((cfg.vocab_size, hidden), 1000),
        f"{layer_prefix}.norm.weight": _make_weight((hidden,), 2000),
        "lm_head.weight": _make_weight((cfg.vocab_size, hidden), 3000),
    }

    for i in range(cfg.num_hidden_layers):
        base = 10_000 * (i + 1)
        p = f"{layer_prefix}.layers.{i}"
        weights[f"{p}.input_layernorm.weight"] = _make_weight((hidden,), base + 1)
        weights[f"{p}.post_attention_layernorm.weight"] = _make_weight((hidden,), base + 101)

        weights[f"{p}.self_attn.q_proj.weight"] = _make_weight((q_size, hidden), base + 1001)
        weights[f"{p}.self_attn.k_proj.weight"] = _make_weight((kv_size, hidden), base + 2001)
        weights[f"{p}.self_attn.v_proj.weight"] = _make_weight((kv_size, hidden), base + 3001)
        weights[f"{p}.self_attn.o_proj.weight"] = _make_weight((hidden, q_size), base + 4001)
        weights[f"{p}.self_attn.q_norm.weight"] = _make_weight((cfg.head_dim,), base + 5001)
        weights[f"{p}.self_attn.k_norm.weight"] = _make_weight((cfg.head_dim,), base + 6001)

        weights[f"{p}.mlp.gate_proj.weight"] = _make_weight((inter, hidden), base + 7001)
        weights[f"{p}.mlp.up_proj.weight"] = _make_weight((inter, hidden), base + 8001)
        weights[f"{p}.mlp.down_proj.weight"] = _make_weight((hidden, inter), base + 9001)

    return weights


def test_load_weights_stacks_qkv_and_gate_up_from_model_prefix():
    cfg = _make_config()
    model = Qwen3ForCausalLM(cfg)

    weights = _build_language_weights(cfg, layer_prefix="model")
    model.load_weights(weights.items())

    layer0 = model.model.layers[0]
    q_size = cfg.num_attention_heads * cfg.head_dim
    kv_size = cfg.num_key_value_heads * cfg.head_dim

    q = weights["model.layers.0.self_attn.q_proj.weight"]
    k = weights["model.layers.0.self_attn.k_proj.weight"]
    v = weights["model.layers.0.self_attn.v_proj.weight"]
    qkv = layer0.self_attn.qkv_proj.weight.data
    assert torch.equal(qkv[:q_size], q)
    assert torch.equal(qkv[q_size : q_size + kv_size], k)
    assert torch.equal(qkv[q_size + kv_size : q_size + 2 * kv_size], v)

    gate = weights["model.layers.0.mlp.gate_proj.weight"]
    up = weights["model.layers.0.mlp.up_proj.weight"]
    gate_up = layer0.mlp.gate_up_proj.weight.data
    assert torch.equal(gate_up[: cfg.intermediate_size], gate)
    assert torch.equal(gate_up[cfg.intermediate_size :], up)

    assert torch.equal(model.model.embed_tokens.weight.data, weights["model.embed_tokens.weight"])
    assert torch.equal(model.model.norm.weight.data, weights["model.norm.weight"])
    assert torch.equal(model.lm_head.weight.data, weights["lm_head.weight"])


def test_load_weights_accepts_model_language_model_prefix():
    cfg = _make_config()
    model = Qwen3ForCausalLM(cfg)

    weights = _build_language_weights(cfg, layer_prefix="model.language_model")
    model.load_weights(weights.items())

    layer1 = model.model.layers[1]
    q = weights["model.language_model.layers.1.self_attn.q_proj.weight"]
    k = weights["model.language_model.layers.1.self_attn.k_proj.weight"]
    v = weights["model.language_model.layers.1.self_attn.v_proj.weight"]

    q_size = cfg.num_attention_heads * cfg.head_dim
    kv_size = cfg.num_key_value_heads * cfg.head_dim
    qkv = layer1.self_attn.qkv_proj.weight.data

    assert torch.equal(qkv[:q_size], q)
    assert torch.equal(qkv[q_size : q_size + kv_size], k)
    assert torch.equal(qkv[q_size + kv_size : q_size + 2 * kv_size], v)


def test_quantized_load_weights_stacks_qkv_and_gate_up_weight_and_scale():
    cfg = _make_config()
    model = Qwen3ForCausalLM(cfg, quant_config=_make_w8a8_config())

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

    q_size = cfg.num_attention_heads * cfg.head_dim
    kv_size = cfg.num_key_value_heads * cfg.head_dim
    hidden = cfg.hidden_size
    inter = cfg.intermediate_size

    weights = {
        "model.layers.0.self_attn.q_proj.weight": _make_int8_weight((q_size, hidden), 1),
        "model.layers.0.self_attn.k_proj.weight": _make_int8_weight((kv_size, hidden), 2),
        "model.layers.0.self_attn.v_proj.weight": _make_int8_weight((kv_size, hidden), 3),
        "model.layers.0.self_attn.q_proj.weight_scale": torch.full((q_size, 1), 0.1),
        "model.layers.0.self_attn.k_proj.weight_scale": torch.full((kv_size, 1), 0.2),
        "model.layers.0.self_attn.v_proj.weight_scale": torch.full((kv_size, 1), 0.3),
        "model.layers.0.mlp.gate_proj.weight": _make_int8_weight((inter, hidden), 4),
        "model.layers.0.mlp.up_proj.weight": _make_int8_weight((inter, hidden), 5),
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
    assert torch.equal(
        gate_up.weight[:inter],
        weights["model.layers.0.mlp.gate_proj.weight"],
    )
    assert torch.equal(
        gate_up.weight[inter : 2 * inter],
        weights["model.layers.0.mlp.up_proj.weight"],
    )
    torch.testing.assert_close(
        gate_up.weight_scale[:inter],
        weights["model.layers.0.mlp.gate_proj.weight_scale"],
    )
    torch.testing.assert_close(
        gate_up.weight_scale[inter : 2 * inter],
        weights["model.layers.0.mlp.up_proj.weight_scale"],
    )
