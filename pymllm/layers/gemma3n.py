from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

from pymllm.layers.base import MllmBaseLayer
from pymllm.layers.linear import Linear
from pymllm.layers.utils import set_weight_attrs


def _get_gemma3n_hidden_act_fn(name: str):
    name = (name or "silu").lower()
    if name in ("silu", "swish"):
        return F.silu
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name in ("gelu_tanh", "gelu_pytorch_tanh"):
        return lambda x: F.gelu(x, approximate="tanh")
    raise ValueError(f"Unsupported Gemma3n activation: {name}")


class Gemma3nRMSNorm(MllmBaseLayer):
    """Gemma3n RMSNorm used by the native text-only implementation.

    This intentionally preserves the numerics of the verified Gemma3n path.
    It is not replaced by the generic ``GemmaRMSNorm`` because the generic
    FlashInfer-backed layer has different behavior for this checkpoint path.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = Parameter(torch.empty(hidden_size))
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
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


class Gemma3nRMSNormNoWeight(MllmBaseLayer):
    """Weight-free RMSNorm used by Gemma3n value normalization."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x_fp32 = x.float()
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        out = x_fp32 * torch.rsqrt(var + self.eps)
        return out.to(x_dtype)


class Gemma3nMLP(MllmBaseLayer):
    """Gemma3n feed-forward block.

    Uses pymllm ``Linear`` projections while preserving Gemma3n's optional
    activation sparsity branch, which is not implemented by the generic
    ``pymllm.layers.MLP``.
    """

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
        self.act = _get_gemma3n_hidden_act_fn(activation)
        self.activation_sparsity = float(activation_sparsity)

        if self.activation_sparsity > 0.0:
            normal_dist = torch.distributions.normal.Normal(0.0, 1.0)
            std_multiplier = normal_dist.icdf(
                torch.tensor(self.activation_sparsity, dtype=torch.float32)
            )
            self.register_buffer(
                "_std_multiplier",
                std_multiplier,
                persistent=False,
            )

        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)

    def _gaussian_topk(self, inputs: torch.Tensor) -> torch.Tensor:
        std_multiplier = self._std_multiplier.to(
            device=inputs.device,
            dtype=inputs.dtype,
        )
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
