import copy
import json
import pytest

from pymllm.executor.model_runner import ModelRunner
from pymllm.quantization import get_quantization_config, list_quantization_methods
from pymllm.quantization.methods.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)


def _current_ct_config():
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "group_size": 32,
                    "strategy": "group",
                    "symmetric": True,
                    "actorder": None,
                },
            },
        },
        "ignore": ["ignore_prefix"],
    }


def _current_ct_w8a8_config():
    return {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
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
            },
        },
        "ignore": ["ignore_prefix"],
    }


def test_compressed_tensors_is_registered():
    assert "compressed-tensors" in list_quantization_methods()
    assert get_quantization_config("compressed-tensors") is CompressedTensorsConfig


def test_from_config_parses_current_signature():
    config = CompressedTensorsConfig.from_config(
        copy.deepcopy(_current_ct_config())
    )

    assert config.quant_format == "pack-quantized"
    assert config.weight_bits == 4
    assert config.group_size == 32
    assert config.symmetric is True
    assert config.actorder is None
    assert config.ignore == ["ignore_prefix"]


def test_from_config_parses_w8a8_signature():
    config = CompressedTensorsConfig.from_config(
        copy.deepcopy(_current_ct_w8a8_config())
    )

    assert config.quant_format == "int-quantized"
    assert config.weight_bits == 8
    assert config.group_size is None
    assert config.weight_strategy == "channel"
    assert config.weight_type == "int"
    assert config.symmetric is True
    assert config.input_bits == 8
    assert config.input_strategy == "token"
    assert config.input_dynamic is True
    assert config.ignore == ["ignore_prefix"]


def test_load_quant_config_dict_unwraps_quantization_config_from_config_json(
    tmp_path,
):
    root_config = {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "quantization_config": copy.deepcopy(_current_ct_config()),
    }
    (tmp_path / "config.json").write_text(json.dumps(root_config))

    loaded = ModelRunner._load_quant_config_dict(tmp_path)

    assert loaded == root_config["quantization_config"]


def test_get_quant_method_respects_ignore():
    config = CompressedTensorsConfig.from_config(
        copy.deepcopy(_current_ct_config())
    )
    assert config.get_quant_method(layer=None, prefix="ignore_prefix.layer") is None

    method = config.get_quant_method(
        layer=None,
        prefix="model.language_model.layers.0.self_attn.q_proj",
    )
    assert isinstance(method, CompressedTensorsLinearMethod)

def test_get_quant_method_rejects_unsupported_signature():
    checkpoint_config = copy.deepcopy(_current_ct_config())
    checkpoint_config["config_groups"]["group_0"]["weights"]["group_size"] = 128

    config = CompressedTensorsConfig.from_config(checkpoint_config)

    with pytest.raises(ValueError, match="group_size"):
        config.get_quant_method(
            layer=None,
            prefix="model.language_model.layers.0.self_attn.q_proj",
        )


def test_get_quant_method_accepts_w8a8_signature():
    config = CompressedTensorsConfig.from_config(
        copy.deepcopy(_current_ct_w8a8_config())
    )
    method = config.get_quant_method(
        layer=None,
        prefix="model.language_model.layers.0.self_attn.q_proj",
    )
    assert isinstance(method, CompressedTensorsLinearMethod)
