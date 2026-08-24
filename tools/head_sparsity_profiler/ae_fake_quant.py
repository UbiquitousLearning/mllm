"""Minimal fake W8A8 quantization used by the ShadowNPU AE profiler.

This is intentionally local to the profiler.  The repository's deployment
converter has evolved to W8A16 Q/K/V simulation, whereas the published AE
profiles all seven attention/MLP projections with this static W8A8 path.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as functional
from torch import nn


def get_ae_clip_and_scale(
    activation_distribution: dict[str, Any], threshold: float = 64.0
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, bool]], dict[str, int]]:
    """Apply the AE's top-0.1%-versus-original clipping decision."""

    top = activation_distribution["top_0_1"]
    original = activation_distribution["ori"]
    statistics = activation_distribution["all_stat"]
    scales: dict[str, dict[str, Any]] = {}
    clipping: dict[str, dict[str, bool]] = {}
    summary = {
        "clip_input_num": 0,
        "no_clip_input_num": 0,
        "clip_output_num": 0,
        "no_clip_output_num": 0,
    }
    for name in statistics:
        scales[name] = {}
        clipping[name] = {}
        for side in ("input", "output"):
            clipped = top[name][side] * threshold > original[name][side]
            clipping[name][side] = clipped
            scales[name][side] = original[name][side] if clipped else top[name][side]
            summary[f"{'clip' if clipped else 'no_clip'}_{side}_num"] += 1
    return scales, clipping, summary


@torch.no_grad()
def _quantize_weight(tensor: torch.Tensor) -> torch.Tensor:
    result = tensor.detach().clone()
    scale = result.abs().max().clamp(min=1e-5) / 127
    result.div_(scale).round_().mul_(scale)
    return result


@torch.no_grad()
def _quantize_input(tensor: torch.Tensor, maximum: torch.Tensor, clip: bool) -> torch.Tensor:
    # ``profile_heads.py`` uses the older AE simulation which rounds input
    # quantization scales to five decimal places.
    scale = maximum.clone().to(tensor.device).clamp(min=1e-5) / 127
    scale = (scale * 100000).round() / 100000
    result = tensor.div(scale).round()
    if clip:
        result = result.clamp(-128.0, 127.0)
    return result.mul(scale)


@torch.no_grad()
def _quantize_output(tensor: torch.Tensor, maximum: torch.Tensor, clip: bool) -> torch.Tensor:
    scale = maximum.clone().to(tensor.device).clamp(min=1e-5) / 127
    result = tensor.div(scale).round()
    if clip:
        result = result.clamp(-128.0, 127.0)
    return result.mul(scale)


class AEW8A8Linear(nn.Module):
    def __init__(
        self,
        source: nn.Linear,
        scales: dict[str, Any],
        clipping: dict[str, bool],
    ) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        # Keep these as ordinary float32 CPU tensors, matching the AE module.
        # Each forward clones them to the activation device; model.to(dtype)
        # must not silently turn static scales into bfloat16.
        self.input_scale = torch.tensor(scales["input"])
        self.output_scale = torch.tensor(scales["output"])
        self.register_buffer("weight", _quantize_weight(source.weight))
        self.register_buffer(
            "bias", _quantize_weight(source.bias) if source.bias is not None else None
        )
        self.clip_input = bool(clipping["input"])
        self.clip_output = bool(clipping["output"])

    @torch.no_grad()
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = _quantize_input(tensor, self.input_scale, self.clip_input)
        tensor = functional.linear(tensor, self.weight, self.bias)
        return _quantize_output(tensor, self.output_scale, self.clip_output)


def quantize_model_ae(
    model: Any,
    layers: Sequence[Any],
    scales: dict[str, dict[str, Any]],
    clipping: dict[str, dict[str, bool]],
) -> int:
    """Replace the seven Qwen2 projections per decoder layer."""

    base = getattr(model, "model", None)
    if base is None:
        raise RuntimeError("AE fake quantization expects model.model")
    names = {id(module): name for name, module in base.named_modules()}
    replacements = 0
    for layer_index, layer in enumerate(layers):
        units = (
            (getattr(layer, "self_attn", None), ("q_proj", "k_proj", "v_proj", "o_proj")),
            (getattr(layer, "mlp", None), ("gate_proj", "up_proj", "down_proj")),
        )
        for unit, attributes in units:
            if unit is None or id(unit) not in names:
                raise RuntimeError(f"layer {layer_index} has an unsupported module layout")
            unit_name = names[id(unit)]
            for attribute in attributes:
                source = getattr(unit, attribute, None)
                if not isinstance(source, nn.Linear):
                    raise RuntimeError(
                        f"layer {layer_index} {unit_name}.{attribute} is not nn.Linear"
                    )
                key = f"model.{unit_name}.{attribute}"
                if key not in scales or key not in clipping:
                    raise RuntimeError(f"activation distribution has no entry for {key}")
                setattr(unit, attribute, AEW8A8Linear(source, scales[key], clipping[key]))
                replacements += 1
    return replacements
