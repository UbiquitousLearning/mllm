"""Quantization method base classes for pymllm layers.

This module defines the plugin interface that all quantization methods must
implement.  The pattern follows sglang / vLLM's ``LinearMethodBase`` design:

1. Each quantization algorithm (AWQ, GPTQ, FP8, ...) provides a concrete
   subclass of :class:`LinearMethodBase`.
2. Linear layers hold a ``quant_method`` attribute (an instance of
   :class:`LinearMethodBase`).
3. During ``__init__``, the linear layer calls
   ``quant_method.create_weights(layer, ...)`` to register the appropriate
   parameters (packed int weights, scales, zero-points, etc.) on itself.
4. During ``forward``, the linear layer calls
   ``quant_method.apply(layer, x, bias)`` instead of ``F.linear``.
5. After checkpoint loading, :class:`~pymllm.executor.model_runner.ModelRunner`
   iterates all modules and calls
   ``quant_method.process_weights_after_loading(layer)`` for format conversion,
   repacking (e.g. AWQ → Marlin), or calibration.

Typical lifecycle::

    # ---- model construction ----
    quant_method = SomeLinearMethod(bits=4, group_size=128)
    layer = ColumnParallelLinear(4096, 4096, quant_method=quant_method)
    #   → calls quant_method.create_weights(layer, ...)
    #   → layer now has .qweight, .scales, .qzeros, etc.

    # ---- weight loading ----
    model.load_weights(iter_weights(...))
    #   → checkpoint tensors are loaded into the parameters created above,
    #     using each parameter's ``weight_loader`` attribute.

    # ---- post-load processing ----
    for module in model.modules():
        qm = getattr(module, "quant_method", None)
        if qm is not None:
            qm.process_weights_after_loading(module)
    #   → AWQ repacks int4 → Marlin layout, GPTQ shuffles by g_idx, etc.

    # ---- inference ----
    output = layer(x)
    #   → calls quant_method.apply(layer, x, bias)
    #   → dequant + matmul (or fused kernel)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from pymllm.layers.utils import set_weight_attrs


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class QuantizeMethodBase(ABC):
    """Base class for all quantization methods (linear, embedding, MoE, ...).

    Every concrete quantization algorithm must implement at least
    :meth:`create_weights` and :meth:`apply`.

    How to implement a new quantization method
    -------------------------------------------
    1. Subclass :class:`LinearMethodBase` (for linear layers).
    2. Override :meth:`create_weights` to register quantized parameters
       (``qweight``, ``scales``, ``qzeros``, etc.) on the layer via
       ``layer.register_parameter()``.
    3. Override :meth:`apply` to perform the quantized forward computation.
    4. Optionally override :meth:`process_weights_after_loading` if the
       checkpoint format differs from the runtime format (e.g. repacking,
       transposing, or calibrating scales).
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Create and register quantized weight parameters on *layer*.

        Called once during layer construction (``__init__``).  Implementations
        should call ``layer.register_parameter(name, param)`` and attach
        metadata via :func:`~pymllm.layers.utils.set_weight_attrs` so that
        the weight-loading infrastructure knows how to shard and load them.

        Parameters
        ----------
        layer
            The ``nn.Module`` (e.g. ``ColumnParallelLinear``) that will own
            the parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute the quantized forward pass.

        Called by ``layer.forward()`` every inference step.  The method should
        read the parameters previously created by :meth:`create_weights` from
        *layer* (e.g. ``layer.qweight``, ``layer.scales``), dequantize or
        invoke a fused kernel, and return the output tensor.

        Parameters
        ----------
        layer
            The module that owns the quantized parameters.
        """
        raise NotImplementedError

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-process parameters after checkpoint loading.

        Called once by ``ModelRunner`` after all checkpoint tensors have been
        loaded into the layer's parameters.  Use this for:

        * **Repacking**: converting checkpoint layout to kernel-native layout
          (e.g. AutoAWQ int4 → Marlin packed format).
        * **Transposing**: rearranging dimensions for optimised GEMM kernels.
        * **Calibration**: computing per-tensor or per-channel scales from
          the loaded FP weights (e.g. dynamic FP8 quantisation).
        * **Cleanup**: replacing custom parameter wrappers with plain
          ``torch.nn.Parameter`` to avoid overhead during inference.

        The default implementation is a no-op.
        """
        return


class LinearMethodBase(QuantizeMethodBase):
    """Base class for quantization methods applied to linear layers.

    Narrows the :class:`QuantizeMethodBase` interface with concrete
    signatures tailored to linear (matmul) operations.

    Subclasses must implement :meth:`create_weights` and :meth:`apply`.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        """Create quantized weight tensors on *layer*.

        Parameters
        ----------
        layer
            The linear module that will own the parameters.
        input_size_per_partition
            Number of input features on this TP rank.
        output_partition_sizes
            Output sizes of each logical weight on this TP rank.  For a
            standard linear layer this is ``[out_features_per_partition]``.
            For a merged QKV layer it might be ``[q_size, k_size, v_size]``.
        input_size
            Full (un-sharded) input dimension.
        output_size
            Full (un-sharded) output dimension.
        params_dtype
            Data type for full-precision parameters (e.g. ``torch.float16``).
        **extra_weight_attrs
            Additional metadata to attach to created parameters (e.g.
            ``weight_loader``, ``packed_dim``, ``packed_factor``).

        Example (AWQ W4A16)::

            # Register packed 4-bit weights, scales, and zero-points
            qweight = Parameter(torch.empty(..., dtype=torch.int32))
            layer.register_parameter("qweight", qweight)

            scales = Parameter(torch.empty(..., dtype=params_dtype))
            layer.register_parameter("scales", scales)

            qzeros = Parameter(torch.empty(..., dtype=torch.int32))
            layer.register_parameter("qzeros", qzeros)
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the quantized linear forward.

        Parameters
        ----------
        layer
            The module that owns quantized parameters (set by
            :meth:`create_weights`).
        x
            Input activation tensor, shape ``(*, input_size_per_partition)``.
        bias
            Optional bias vector.

        Returns
        -------
        torch.Tensor
            Output tensor, shape ``(*, sum(output_partition_sizes))``.

        Example (AWQ W4A16)::

            qweight = layer.qweight   # packed int32
            scales  = layer.scales     # fp16 per-group scales
            qzeros  = layer.qzeros     # packed int32 zero-points
            # → invoke dequant + matmul kernel
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Default unquantized implementation
# ---------------------------------------------------------------------------


class UnquantizedLinearMethod(LinearMethodBase):
    """Default pass-through for non-quantized linear layers.

    Creates a standard FP weight ``(out_features, in_features)`` and
    forwards via ``F.linear``.  This is used when no quantization config
    is specified so that every linear layer always has a ``quant_method``
    attribute with a uniform interface.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        """Create a standard full-precision weight parameter."""
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard ``F.linear`` forward."""
        return F.linear(x, layer.weight, bias)
