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


def _current_ct_w8a8_config() -> dict:
    return {
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


def _build_quant_method_w8a8() -> ct.CompressedTensorsLinearMethod:
    cfg = ct.CompressedTensorsConfig.from_config(_current_ct_w8a8_config())
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


def test_w8a8_create_weights_registers_weight_and_scale():
    layer = _DummyLayer()
    qm = _build_quant_method_w8a8()

    qm.create_weights(
        layer=layer,
        input_size_per_partition=64,
        output_partition_sizes=[96],
        input_size=64,
        output_size=96,
        params_dtype=torch.float16,
        weight_loader=_weight_loader,
    )

    assert {"weight", "weight_scale"} <= set(layer._parameters)
    assert tuple(layer.weight.shape) == (96, 64)
    assert layer.weight.dtype == torch.int8
    assert tuple(layer.weight_scale.shape) == (96, 1)
    assert layer.weight_scale.dtype == torch.float32


def test_w8a8_process_weights_transposes_and_flattens_scales():
    layer = _DummyLayer()
    qm = _build_quant_method_w8a8()
    qm.create_weights(
        layer=layer,
        input_size_per_partition=32,
        output_partition_sizes=[48],
        input_size=32,
        output_size=48,
        params_dtype=torch.float16,
        weight_loader=_weight_loader,
    )

    with torch.no_grad():
        layer.weight.copy_(
            torch.arange(layer.weight.numel(), dtype=torch.int8).reshape_as(layer.weight)
        )
        layer.weight_scale.copy_(
            torch.arange(1, 49, dtype=torch.float32).reshape(48, 1) / 100.0
        )

    qm.process_weights_after_loading(layer)

    assert tuple(layer.weight.shape) == (32, 48)
    assert layer.weight.is_contiguous()
    assert tuple(layer.weight_scale.shape) == (48,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w8a8_apply_matches_reference_for_large_m():
    layer = _DummyLayer()
    qm = _build_quant_method_w8a8()

    with torch.device("cuda"):
        qm.create_weights(
            layer=layer,
            input_size_per_partition=64,
            output_partition_sizes=[128],
            input_size=64,
            output_size=128,
            params_dtype=torch.float16,
            weight_loader=_weight_loader,
        )

    with torch.no_grad():
        layer.weight.copy_(
            torch.randint(-127, 128, layer.weight.shape, device="cuda", dtype=torch.int8)
        )
        layer.weight_scale.copy_(
            torch.rand(layer.weight_scale.shape, device="cuda", dtype=torch.float32)
            + 1e-3
        )
    qm.process_weights_after_loading(layer)

    x = torch.randn(32, 64, device="cuda", dtype=torch.float16)
    bias = torch.randn(128, device="cuda", dtype=torch.float16)
    out = qm.apply(layer, x, bias)

    x_q, x_scale = ct._per_token_quant_int8(x)
    ref_i32 = torch._int_mm(x_q, layer.weight).to(torch.float32)
    ref = (ref_i32 * x_scale * layer.weight_scale.view(1, -1)).to(x.dtype) + bias

    assert out.shape == (32, 128)
    assert torch.allclose(out, ref, atol=2e-1, rtol=2e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w8a8_apply_supports_small_m_by_padding():
    layer = _DummyLayer()
    qm = _build_quant_method_w8a8()

    with torch.device("cuda"):
        qm.create_weights(
            layer=layer,
            input_size_per_partition=64,
            output_partition_sizes=[64],
            input_size=64,
            output_size=64,
            params_dtype=torch.float16,
            weight_loader=_weight_loader,
        )

    with torch.no_grad():
        layer.weight.copy_(
            torch.randint(-127, 128, layer.weight.shape, device="cuda", dtype=torch.int8)
        )
        layer.weight_scale.fill_(0.01)
    qm.process_weights_after_loading(layer)

    x = torch.randn(2, 64, device="cuda", dtype=torch.float16)
    out = qm.apply(layer, x)

    assert out.shape == (2, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_w8a8_apply_uses_triton_quant_and_torch_int_mm(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify the W8A8 forward path uses Triton activation quant + torch._int_mm."""
    layer = _DummyLayer()
    qm = _build_quant_method_w8a8()

    with torch.device("cuda"):
        qm.create_weights(
            layer=layer,
            input_size_per_partition=64,
            output_partition_sizes=[64],
            input_size=64,
            output_size=64,
            params_dtype=torch.float16,
            weight_loader=_weight_loader,
        )

    with torch.no_grad():
        layer.weight.copy_(
            torch.randint(-127, 128, layer.weight.shape, device="cuda", dtype=torch.int8)
        )
        layer.weight_scale.fill_(0.01)
    qm.process_weights_after_loading(layer)

    # Track that Triton quantization is called
    triton_quant_calls: list[tuple] = []
    original_triton_quant = None
    try:
        from pymllm.quantization.kernels.int8_activation_triton import (
            per_token_quant_int8 as _original,
        )
        original_triton_quant = _original
    except ImportError:
        pass

    def tracked_triton_quant(x, **kwargs):
        triton_quant_calls.append(tuple(x.shape))
        return original_triton_quant(x, **kwargs)

    import pymllm.quantization.kernels.int8_activation_triton as triton_mod
    monkeypatch.setattr(triton_mod, "per_token_quant_int8", tracked_triton_quant)

    x = torch.randn(2, 64, device="cuda", dtype=torch.float16)
    bias = torch.randn(64, device="cuda", dtype=torch.float16)
    out = qm.apply(layer, x, bias)

    assert out.shape == (2, 64)
    assert len(triton_quant_calls) == 1, "Triton quant should be called exactly once"
    assert triton_quant_calls[0] == (2, 64)
