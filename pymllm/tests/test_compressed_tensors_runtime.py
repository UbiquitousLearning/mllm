from __future__ import annotations

import pytest
import torch
from torch import nn

import pymllm.quantization.methods.compressed_tensors as ct


def _current_ct_config() -> dict:
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "ignore": ["lm_head"],
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "group_size": 32,
                    "strategy": "group",
                    "symmetric": True,
                    "actorder": None,
                    "type": "int",
                },
            }
        },
    }


class _DummyLayer(nn.Module):
    pass


def _build_quant_method() -> ct.CompressedTensorsLinearMethod:
    cfg = ct.CompressedTensorsConfig.from_config(_current_ct_config())
    qm = cfg.get_quant_method(
        layer=None,
        prefix="model.language_model.layers.0.self_attn.q_proj",
    )
    assert isinstance(qm, ct.CompressedTensorsLinearMethod)
    return qm


def _weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_create_weights_registers_checkpoint_parameter_names():
    layer = _DummyLayer()
    qm = _build_quant_method()

    with torch.device("cuda"):
        qm.create_weights(
            layer=layer,
            input_size_per_partition=2048,
            output_partition_sizes=[2048],
            input_size=2048,
            output_size=2048,
            params_dtype=torch.bfloat16,
            weight_loader=_weight_loader,
        )

    assert {"weight_packed", "weight_scale", "weight_shape"} <= set(
        layer._parameters
    )
    assert tuple(layer.weight_packed.shape) == (2048, 256)
    assert tuple(layer.weight_scale.shape) == (2048, 64)
    assert tuple(layer.weight_shape.shape) == (2,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_process_and_apply_use_gptq_repack_and_uint4b8(
    monkeypatch: pytest.MonkeyPatch,
):
    layer = _DummyLayer()
    qm = _build_quant_method()

    with torch.device("cuda"):
        qm.create_weights(
            layer=layer,
            input_size_per_partition=2048,
            output_partition_sizes=[2048],
            input_size=2048,
            output_size=2048,
            params_dtype=torch.bfloat16,
            weight_loader=_weight_loader,
        )

    with torch.no_grad():
        layer.weight_packed.copy_(
            torch.arange(
                layer.weight_packed.numel(),
                device="cuda",
                dtype=torch.int32,
            ).reshape_as(layer.weight_packed)
        )
        layer.weight_scale.fill_(1)
        layer.weight_shape.copy_(
            torch.tensor([2048, 2048], device="cuda", dtype=torch.int64)
        )

    repack_calls: dict[str, object] = {}
    scale_calls: dict[str, object] = {}
    workspace = torch.zeros(1, dtype=torch.int32, device="cuda")
    empty_tensors: list[torch.Tensor] = []

    monkeypatch.setattr(ct, "verify_marlin_supports_shape", lambda **_: None)
    monkeypatch.setattr(
        ct,
        "marlin_make_workspace",
        lambda device: workspace,
    )
    monkeypatch.setattr(
        ct,
        "marlin_make_empty_g_idx",
        lambda device: empty_tensors.append(
            torch.empty(0, dtype=torch.int32, device=device)
        )
        or empty_tensors[-1],
    )
    monkeypatch.setattr(
        ct,
        "gptq_marlin_repack",
        lambda b_q_weight, perm, size_k, size_n, num_bits: repack_calls.update(
            {
                "b_q_weight": b_q_weight,
                "perm": perm,
                "size_k": size_k,
                "size_n": size_n,
                "num_bits": num_bits,
            }
        )
        or torch.zeros(
            (size_k // 16, size_n * 16 // (32 // num_bits)),
            dtype=torch.int32,
            device=b_q_weight.device,
        ),
    )
    monkeypatch.setattr(
        ct,
        "marlin_permute_scales",
        lambda s, size_k, size_n, group_size: scale_calls.update(
            {
                "s": s,
                "size_k": size_k,
                "size_n": size_n,
                "group_size": group_size,
            }
        )
        or torch.zeros(
            (size_k // group_size, size_n),
            dtype=s.dtype,
            device=s.device,
        ),
    )

    calls: dict[str, object] = {}

    def fake_gemm(**kwargs):
        calls.update(kwargs)
        return torch.zeros(
            (kwargs["size_m"], kwargs["size_n"]),
            dtype=kwargs["a"].dtype,
            device=kwargs["a"].device,
        )

    monkeypatch.setattr(ct, "gptq_marlin_gemm", fake_gemm)

    qm.process_weights_after_loading(layer)
    x = torch.randn(2, 2048, device="cuda", dtype=torch.bfloat16)
    out = qm.apply(layer, x)

    assert out.shape == (2, 2048)
    assert repack_calls["perm"] is layer.g_idx_sort_indices
    assert repack_calls["size_k"] == 2048
    assert repack_calls["size_n"] == 2048
    assert repack_calls["num_bits"] == 4
    assert scale_calls["size_k"] == 2048
    assert scale_calls["size_n"] == 2048
    assert scale_calls["group_size"] == 32
    assert calls["workspace"] is workspace
    assert calls["b_zeros"] is layer.weight_zero_point
    assert calls["g_idx"] is layer.weight_g_idx
    assert calls["perm"] is layer.g_idx_sort_indices
    assert calls["b_q_type_id"] == ct.SCALAR_TYPE_UINT4B8.id
    assert calls["b_q_weight"] is layer.weight_packed
