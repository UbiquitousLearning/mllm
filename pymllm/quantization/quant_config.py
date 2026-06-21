"""Quantization configuration base class and registry.

This module provides the bridge between a model checkpoint's quantization
metadata (e.g. ``quantize_config.json``) and the runtime
:class:`~pymllm.layers.quantize_base.LinearMethodBase` instances used by
each linear layer.

Architecture overview::

    quantize_config.json   ──parse──►  QuantizationConfig subclass
                                          │
                                          │  get_quant_method(layer, prefix)
                                          ▼
                                     LinearMethodBase instance
                                      (AWQLinearMethod, FP8LinearMethod, ...)

How to add a new quantization method
-------------------------------------
1. Create a ``QuantizationConfig`` subclass (e.g. ``AWQConfig``).
2. Implement ``get_name()``, ``from_config()``, ``get_quant_method()``.
3. Register it::

       from pymllm.quantization.quant_config import register_quantization

       @register_quantization("awq")
       class AWQConfig(QuantizationConfig):
           ...

4. When the server starts with ``--quantization.method awq``, the loader
   will call ``get_quantization_config("awq")`` to obtain the config class,
   then ``from_config(hf_quant_config)`` to instantiate it, and finally
   ``config.get_quant_method(layer, prefix)`` for each linear layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import torch

from pymllm.layers.quantize_base import QuantizeMethodBase


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps method name (e.g. "awq", "gptq", "fp8") to config class.
_QUANTIZATION_REGISTRY: Dict[str, Type[QuantizationConfig]] = {}


def register_quantization(
    name: str,
) -> "type[type[QuantizationConfig]]":
    """Class decorator that registers a :class:`QuantizationConfig` subclass.

    Usage::

        @register_quantization("awq")
        class AWQConfig(QuantizationConfig):
            ...
    """

    def decorator(cls: Type[QuantizationConfig]) -> Type[QuantizationConfig]:
        if name in _QUANTIZATION_REGISTRY:
            raise ValueError(
                f"Quantization method {name!r} is already registered "
                f"by {_QUANTIZATION_REGISTRY[name].__name__}"
            )
        _QUANTIZATION_REGISTRY[name] = cls
        return cls

    return decorator  # type: ignore[return-value]


def get_quantization_config(method: str) -> Type[QuantizationConfig]:
    """Look up a registered :class:`QuantizationConfig` by name.

    Raises ``KeyError`` if the method is not registered.
    """
    if method not in _QUANTIZATION_REGISTRY:
        supported = ", ".join(sorted(_QUANTIZATION_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown quantization method {method!r}. "
            f"Registered methods: {supported}"
        )
    return _QUANTIZATION_REGISTRY[method]


def list_quantization_methods() -> List[str]:
    """Return sorted list of registered quantization method names."""
    return sorted(_QUANTIZATION_REGISTRY)


# ---------------------------------------------------------------------------
# Base config class
# ---------------------------------------------------------------------------


class QuantizationConfig(ABC):
    """Base class for quantization configurations.

    A ``QuantizationConfig`` is instantiated once per model load.  It reads
    quantization metadata from the checkpoint (bit-width, group size, etc.)
    and provides :class:`~pymllm.layers.quantize_base.QuantizeMethodBase`
    instances to each layer.

    Subclass contract
    -----------------
    * :meth:`get_name` — return the method name (e.g. ``"awq"``).
    * :meth:`from_config` — class method that parses a dict from the
      checkpoint's ``quantize_config.json``.
    * :meth:`get_quant_method` — return the appropriate
      ``LinearMethodBase`` (or ``None`` to skip quantization for a layer).

    Optional overrides
    ------------------
    * :meth:`get_supported_act_dtypes` — restrict activation dtypes.
    * :meth:`get_min_capability` — minimum GPU compute capability.
    * :meth:`get_config_filenames` — files to probe in the checkpoint dir.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Return the canonical name of this quantization method.

        Examples: ``"awq"``, ``"gptq"``, ``"fp8"``, ``"w8a8"``.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig":
        """Create an instance from a checkpoint's quantization config dict.

        Parameters
        ----------
        config
            Parsed JSON from the checkpoint's ``quantize_config.json`` or
            the ``quantization_config`` section of ``config.json``.

        Example config dict (AWQ)::

            {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "zero_point": true
            }
        """
        raise NotImplementedError

    @abstractmethod
    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str = "",
    ) -> Optional[QuantizeMethodBase]:
        """Return the quantization method for *layer*, or ``None`` to skip.

        Parameters
        ----------
        layer
            The ``nn.Module`` being constructed (e.g. ``ColumnParallelLinear``).
        prefix
            The layer's full dotted name in the model (e.g.
            ``"model.layers.0.self_attn.q_proj"``).  Can be used to
            selectively skip quantization for certain layers.

        Returns
        -------
        QuantizeMethodBase or None
            The method instance.  ``None`` means this layer should fall back
            to the default :class:`~pymllm.layers.quantize_base.UnquantizedLinearMethod`.
        """
        raise NotImplementedError

    # -- Optional hooks (with sensible defaults) --

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        """Activation dtypes supported by this method.

        Override to restrict (e.g. FP8 only supports ``float16``).
        Default: no restriction.
        """
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        """Minimum CUDA compute capability (e.g. 75 for Turing).

        Default: 0 (no restriction).
        """
        return 0

    @staticmethod
    def get_config_filenames() -> List[str]:
        """File names to look for in the checkpoint directory.

        Default: ``["quantize_config.json"]``.
        """
        return ["quantize_config.json"]
