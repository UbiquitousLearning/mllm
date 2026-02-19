from __future__ import annotations

import logging
from typing import Callable, Literal, Optional

import flashinfer
import torch

from pymllm.layers.base import MllmBaseLayer
from pymllm.layers.linear import ColumnParallelLinear, Linear, RowParallelLinear

logger = logging.getLogger(__name__)

MLPActivation = Literal["silu", "gelu", "gelu_tanh"]

_ACTIVATION_MAP: dict[MLPActivation, Callable[..., torch.Tensor]] = {
    "silu": flashinfer.activation.silu_and_mul,
    "gelu": flashinfer.activation.gelu_and_mul,
    "gelu_tanh": flashinfer.activation.gelu_tanh_and_mul,
}


def _validate_mlp_args(
    hidden_size: int, intermediate_size: int, activation: str
) -> None:
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be > 0, but got {hidden_size}")
    if intermediate_size <= 0:
        raise ValueError(
            f"intermediate_size must be > 0, but got {intermediate_size}"
        )
    if activation not in _ACTIVATION_MAP:
        raise ValueError(
            f"Unsupported activation '{activation}'. "
            f"Expected one of: {list(_ACTIVATION_MAP)}"
        )


def _run_gated_activation(
    gate_up: torch.Tensor,
    intermediate_size: int,
    activation: MLPActivation,
    enable_pdl: Optional[bool],
) -> torch.Tensor:
    if gate_up.shape[-1] != 2 * intermediate_size:
        raise ValueError(
            "Expected last dim of gate_up tensor to be "
            f"{2 * intermediate_size}, but got {gate_up.shape[-1]}"
        )
    return _ACTIVATION_MAP[activation](gate_up, enable_pdl=enable_pdl)


class MLP(MllmBaseLayer):
    """Feed-forward MLP block with FlashInfer fused gated activations.

    Non-parallel version (TP=1). Uses :class:`Linear` for all projections.

    Supported activations: ``silu``, ``gelu``, ``gelu_tanh``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: MLPActivation = "silu",
        use_fused_gate_up_proj: bool = True,
        use_bias_gate_up: bool = False,
        use_bias_down: bool = False,
        enable_pdl: Optional[bool] = None,
    ):
        super().__init__()
        _validate_mlp_args(hidden_size, intermediate_size, activation)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.use_fused_gate_up_proj = use_fused_gate_up_proj
        self.enable_pdl = enable_pdl

        if not use_fused_gate_up_proj:
            logger.warning(
                "MLP with use_fused_gate_up_proj=False uses a lower-efficiency path. "
                "Use use_fused_gate_up_proj=True for better performance.",
            )

        if use_fused_gate_up_proj:
            self.gate_up_proj = Linear(
                hidden_size, 2 * intermediate_size, bias=use_bias_gate_up,
            )
            self.gate_proj = None
            self.up_proj = None
        else:
            self.gate_up_proj = None
            self.gate_proj = Linear(
                hidden_size, intermediate_size, bias=use_bias_gate_up,
            )
            self.up_proj = Linear(
                hidden_size, intermediate_size, bias=use_bias_gate_up,
            )

        self.down_proj = Linear(
            intermediate_size, hidden_size, bias=use_bias_down,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected last dim == hidden_size ({self.hidden_size}), "
                f"but got input shape {tuple(x.shape)}"
            )

        if self.use_fused_gate_up_proj:
            assert self.gate_up_proj is not None
            gate_up = self.gate_up_proj(x)
        else:
            assert self.gate_proj is not None and self.up_proj is not None
            gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)

        hidden = _run_gated_activation(
            gate_up, self.intermediate_size, self.activation, self.enable_pdl,
        )
        return self.down_proj(hidden)


class ParallelMLP(MllmBaseLayer):
    """Tensor-parallel MLP with column-sharded intermediate dimension.

    Projection layout (Megatron-style):

    - ``gate_proj``: :class:`ColumnParallelLinear`
      ``(hidden_size → intermediate_size, gather_output=False)``
    - ``up_proj``: :class:`ColumnParallelLinear`
      ``(hidden_size → intermediate_size, gather_output=False)``
    - ``down_proj``: :class:`RowParallelLinear`
      ``(intermediate_size → hidden_size, reduce_output=True)``

    Gate and up projections are kept separate so that each TP rank holds a
    correctly paired ``[gate_shard, up_shard]`` for the gated activation.

    Cost: **1 all-reduce** (inside ``down_proj``).

    Input shape : ``(*, hidden_size)``  — full / replicated.
    Output shape: ``(*, hidden_size)``  — full / replicated.

    Args:
        hidden_size: Model hidden dimension.
        intermediate_size: Intermediate (expanded) dimension **before** TP
            sharding.
        activation: Gated activation type.
        use_bias_gate_up: Add bias to the gate/up projections.
        use_bias_down: Add bias to the down projection.
        enable_pdl: FlashInfer PDL flag.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: MLPActivation = "silu",
        use_bias_gate_up: bool = False,
        use_bias_down: bool = False,
        enable_pdl: Optional[bool] = None,
    ):
        super().__init__()
        _validate_mlp_args(hidden_size, intermediate_size, activation)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.enable_pdl = enable_pdl

        self.gate_proj = ColumnParallelLinear(
            hidden_size, intermediate_size,
            bias=use_bias_gate_up, gather_output=False,
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size, intermediate_size,
            bias=use_bias_gate_up, gather_output=False,
        )

        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size,
            bias=use_bias_down, reduce_output=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected last dim == hidden_size ({self.hidden_size}), "
                f"but got input shape {tuple(x.shape)}"
            )

        gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)

        shard_inter = self.down_proj.in_features_per_partition
        hidden = _run_gated_activation(
            gate_up, shard_inter, self.activation, self.enable_pdl,
        )
        return self.down_proj(hidden)
