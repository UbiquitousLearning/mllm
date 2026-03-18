"""Linear layers with quantization method dispatch.

Every linear layer holds a ``quant_method`` attribute (an instance of
:class:`~pymllm.layers.quantize_base.LinearMethodBase`).  When no
quantization is configured, :class:`UnquantizedLinearMethod` is used as the
default — it creates a standard FP weight and forwards via ``F.linear``.

Quantized checkpoints plug in a different ``LinearMethodBase`` (e.g.
``AWQLinearMethod``) which creates packed int4 weights, scales, and
zero-points, and overrides :meth:`apply` with a fused dequant+matmul kernel.

Usage in model definitions::

    # Non-quantized (default)
    layer = ColumnParallelLinear(4096, 4096)

    # Quantized — pass a quant_method from QuantizationConfig
    qm = awq_config.get_quant_method(layer, prefix="model.layers.0.q_proj")
    layer = ColumnParallelLinear(4096, 4096, quant_method=qm)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from pymllm.layers.base import MllmBaseLayer
from pymllm.layers.quantize_base import LinearMethodBase, UnquantizedLinearMethod
from pymllm.layers.utils import set_weight_attrs
from pymllm.orchestrator import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)


class ColumnParallelLinear(MllmBaseLayer):
    """Linear layer with column parallelism (output-dimension sharding).

    The weight matrix is split along the output dimension across TP ranks.
    Each rank holds ``out_features / tp_size`` rows of the weight.

    Parameters
    ----------
    in_features
        Size of each input sample.
    out_features
        Size of each output sample (before sharding).
    bias
        If ``True``, adds a learnable bias.
    gather_output
        If ``True``, all-gather the output across TP ranks so every rank
        gets the full ``out_features``.  Set to ``False`` when the next
        layer is a :class:`RowParallelLinear` that expects a split input.
    quant_method
        Quantization method instance.  ``None`` → :class:`UnquantizedLinearMethod`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        quant_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output

        if out_features % self.tp_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by "
                f"tp_size ({self.tp_size})"
            )
        self.out_features_per_partition = divide(out_features, self.tp_size)

        self.output_start_index = self.tp_rank * self.out_features_per_partition
        self.output_end_index = self.output_start_index + self.out_features_per_partition

        # --- Quantization method ---
        # The quant_method creates the weight parameters on this layer via
        # create_weights().  For UnquantizedLinearMethod this creates a
        # standard FP Parameter named "weight".  For quantized methods it
        # may instead create qweight, scales, qzeros, etc.
        self.quant_method = quant_method or UnquantizedLinearMethod()
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=in_features,
            output_partition_sizes=[self.out_features_per_partition],
            input_size=in_features,
            output_size=out_features,
            params_dtype=torch.get_default_dtype(),
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias_flag = True
            self.bias = Parameter(torch.empty(self.out_features_per_partition))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.bias_flag = False
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        """Load sharded weights into the parameter.

        Args:
            param: The parameter to load weights into.
            loaded_weight: The weight tensor loaded from checkpoint (full size).
        """
        output_dim = getattr(param, "output_dim", None)

        if output_dim is None or self.tp_size == 1:
            assert param.data.shape == loaded_weight.shape, (
                f"Shape mismatch: param {param.data.shape} vs "
                f"loaded {loaded_weight.shape}"
            )
            param.data.copy_(loaded_weight)
        else:
            shard_weight = loaded_weight.narrow(
                output_dim,
                self.output_start_index,
                self.out_features_per_partition,
            )
            assert param.data.shape == shard_weight.shape, (
                f"Shard shape mismatch: param {param.data.shape} vs "
                f"shard {shard_weight.shape}"
            )
            param.data.copy_(shard_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Delegate computation to the quant_method.  For unquantized layers
        # this is F.linear; for quantized layers it's a fused dequant+matmul.
        output = self.quant_method.apply(self, x, self.bias)

        if self.gather_output and self.tp_size > 1:
            output = tensor_model_parallel_all_gather(output, dim=-1)

        return output


class RowParallelLinear(MllmBaseLayer):
    """Linear layer with row parallelism (input-dimension sharding).

    The weight matrix is split along the input dimension across TP ranks.
    Each rank holds all ``out_features`` rows but only
    ``in_features / tp_size`` columns.

    Typically placed after a :class:`ColumnParallelLinear` whose
    ``gather_output=False``, so the input is already split.

    Parameters
    ----------
    in_features
        Size of each input sample (before sharding).
    out_features
        Size of each output sample.
    bias
        If ``True``, adds a learnable bias (applied after all-reduce).
    reduce_output
        If ``True``, all-reduce the output across TP ranks.
    quant_method
        Quantization method instance.  ``None`` → :class:`UnquantizedLinearMethod`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        reduce_output: bool = True,
        quant_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()

        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

        self.in_features = in_features
        self.out_features = out_features
        self.reduce_output = reduce_output

        if in_features % self.tp_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by "
                f"tp_size ({self.tp_size})"
            )
        self.in_features_per_partition = divide(in_features, self.tp_size)

        self.input_start_index = self.tp_rank * self.in_features_per_partition
        self.input_end_index = self.input_start_index + self.in_features_per_partition

        # --- Quantization method ---
        self.quant_method = quant_method or UnquantizedLinearMethod()
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.in_features_per_partition,
            output_partition_sizes=[out_features],
            input_size=in_features,
            output_size=out_features,
            params_dtype=torch.get_default_dtype(),
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias_flag = True
            self.bias = Parameter(torch.empty(out_features))
            set_weight_attrs(self.bias, {"weight_loader": self.weight_loader})
        else:
            self.bias_flag = False
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        """Load sharded weights into the parameter.

        Args:
            param: The parameter to load weights into.
            loaded_weight: The weight tensor loaded from checkpoint (full size).
        """
        input_dim = getattr(param, "input_dim", None)

        if input_dim is None or self.tp_size == 1:
            assert param.data.shape == loaded_weight.shape, (
                f"Shape mismatch: param {param.data.shape} vs "
                f"loaded {loaded_weight.shape}"
            )
            param.data.copy_(loaded_weight)
        else:
            shard_weight = loaded_weight.narrow(
                input_dim,
                self.input_start_index,
                self.in_features_per_partition,
            )
            assert param.data.shape == shard_weight.shape, (
                f"Shard shape mismatch: param {param.data.shape} vs "
                f"shard {shard_weight.shape}"
            )
            param.data.copy_(shard_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Delegate computation to the quant_method (no bias here; bias is
        # added after the all-reduce below).
        output = self.quant_method.apply(self, x)

        if self.reduce_output and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)

        if self.bias is not None:
            output = output + self.bias

        return output


class Linear(MllmBaseLayer):
    """Non-parallel linear layer with quantization dispatch.

    Parameters
    ----------
    in_features
        Size of each input sample.
    out_features
        Size of each output sample.
    bias
        If ``True``, adds a learnable bias.
    quant_method
        Quantization method instance.  ``None`` → :class:`UnquantizedLinearMethod`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # --- Quantization method ---
        self.quant_method = quant_method or UnquantizedLinearMethod()
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=in_features,
            output_partition_sizes=[out_features],
            input_size=in_features,
            output_size=out_features,
            params_dtype=torch.get_default_dtype(),
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = Parameter(torch.empty(out_features))
            set_weight_attrs(self.bias, {"weight_loader": self.weight_loader})
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, x, self.bias)
