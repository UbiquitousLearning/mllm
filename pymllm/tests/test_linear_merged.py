from __future__ import annotations

import torch
from torch import nn
from torch.nn import Parameter

from pymllm.layers.linear import MergedLinear
from pymllm.layers.quantize_base import LinearMethodBase
from pymllm.layers.utils import set_weight_attrs
from pymllm.quantization.methods.compressed_tensors import CompressedTensorsConfig


def _w8a8_config() -> CompressedTensorsConfig:
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


def test_merged_linear_weight_loader_stacks_w8a8_qkv_weight_and_scale():
    qm = _w8a8_config().get_quant_method(
        layer=None,
        prefix="model.layers.0.self_attn.qkv_proj",
    )
    layer = MergedLinear(4, [6, 2, 2], bias=False, quant_method=qm)

    q = torch.full((6, 4), 1, dtype=torch.int8)
    k = torch.full((2, 4), 2, dtype=torch.int8)
    v = torch.full((2, 4), 3, dtype=torch.int8)

    layer.weight_loader(layer.weight, q, "q")
    layer.weight_loader(layer.weight, k, "k")
    layer.weight_loader(layer.weight, v, "v")

    assert torch.equal(layer.weight[:6], q)
    assert torch.equal(layer.weight[6:8], k)
    assert torch.equal(layer.weight[8:10], v)

    q_scale = torch.full((6, 1), 0.1, dtype=torch.float32)
    k_scale = torch.full((2, 1), 0.2, dtype=torch.float32)
    v_scale = torch.full((2, 1), 0.3, dtype=torch.float32)

    layer.weight_loader(layer.weight_scale, q_scale, "q")
    layer.weight_loader(layer.weight_scale, k_scale, "k")
    layer.weight_loader(layer.weight_scale, v_scale, "v")

    torch.testing.assert_close(layer.weight_scale[:6], q_scale)
    torch.testing.assert_close(layer.weight_scale[6:8], k_scale)
    torch.testing.assert_close(layer.weight_scale[8:10], v_scale)


class _PackedOutputMethod(LinearMethodBase):
    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        qweight = Parameter(
            torch.empty(
                input_size_per_partition,
                sum(output_partition_sizes) // 8,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(qweight, {"output_dim": 1, **extra_weight_attrs})
        layer.register_parameter("qweight", qweight)
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer, bias
        return x


def test_merged_linear_weight_loader_stacks_packed_output_dim_by_loaded_width():
    layer = MergedLinear(4, [16, 8, 8], bias=False, quant_method=_PackedOutputMethod())

    q = torch.full((4, 2), 1, dtype=torch.int32)
    k = torch.full((4, 1), 2, dtype=torch.int32)
    v = torch.full((4, 1), 3, dtype=torch.int32)

    layer.weight_loader(layer.qweight, q, "q")
    layer.weight_loader(layer.qweight, k, "k")
    layer.weight_loader(layer.qweight, v, "v")

    assert torch.equal(layer.qweight[:, :2], q)
    assert torch.equal(layer.qweight[:, 2:3], k)
    assert torch.equal(layer.qweight[:, 3:4], v)
