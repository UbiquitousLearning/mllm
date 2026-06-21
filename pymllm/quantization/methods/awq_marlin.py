"""AWQ quantization with Marlin kernel acceleration.

This module implements the AWQ Marlin quantization plugin for pymllm,
providing high-performance W4A16 inference via the Marlin GEMM kernel.

Classes
-------
AWQMarlinConfig
    Quantization configuration parsed from ``quantize_config.json``.
AWQMarlinLinearMethod
    Linear method that uses AWQ weight format with Marlin kernel dispatch.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy
import torch
from torch.nn import Parameter

from pymllm.layers.quantize_base import LinearMethodBase
from pymllm.layers.utils import set_weight_attrs
from pymllm.quantization.quant_config import QuantizationConfig, register_quantization

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Marlin constants
# ---------------------------------------------------------------------------

MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_TILE = 16


# ---------------------------------------------------------------------------
# ScalarType helpers (matching host::ScalarType in scalar_type.hpp)
# ---------------------------------------------------------------------------

class _ScalarTypeInfo:
    """Lightweight Python mirror of host::ScalarType for type id computation."""

    def __init__(self, name: str, size_bits: int, type_id: int):
        self.name = name
        self.size_bits = size_bits
        self.id = type_id

    def __repr__(self) -> str:
        return f"ScalarType({self.name})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _ScalarTypeInfo):
            return self.id == other.id
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.id)


def _compute_scalar_type_id(
    exponent: int, mantissa: int, signed: bool, bias: int,
    finite_values_only: bool = False, nan_repr: int = 1,
) -> int:
    """Compute the ScalarType::Id matching the C++ implementation."""
    bit_offset = 0
    result = 0

    for value, width in [
        (exponent, 8),
        (mantissa, 8),
        (signed, 1),
        (bias, 32),
        (finite_values_only, 1),
        (nan_repr, 8),
    ]:
        int_val = int(value)
        mask = (1 << width) - 1
        result |= (int_val & mask) << bit_offset
        bit_offset += width

    return result


# Pre-compute the scalar type ids we need
_uint4_id = _compute_scalar_type_id(0, 4, False, 0)
_uint8_id = _compute_scalar_type_id(0, 8, False, 0)
_uint4b8_id = _compute_scalar_type_id(0, 4, False, 8)
_uint8b128_id = _compute_scalar_type_id(0, 8, False, 128)

SCALAR_TYPE_UINT4 = _ScalarTypeInfo("uint4", 4, _uint4_id)
SCALAR_TYPE_UINT8 = _ScalarTypeInfo("uint8", 8, _uint8_id)


# num_bits -> ScalarType mapping
_TYPE_MAP: Dict[int, _ScalarTypeInfo] = {
    4: SCALAR_TYPE_UINT4,
    8: SCALAR_TYPE_UINT8,
}


# ---------------------------------------------------------------------------
# Marlin utility functions
# ---------------------------------------------------------------------------

def verify_marlin_supported(
    quant_type: _ScalarTypeInfo, group_size: int, has_zp: bool
) -> None:
    """Verify that the Marlin kernel supports this configuration."""
    major, minor = torch.cuda.get_device_capability()
    capability = major * 10 + minor
    if capability < 80:
        raise ValueError(
            f"Marlin requires SM80+ (Ampere or newer). Got SM{capability}."
        )
    if group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
        raise ValueError(
            f"Marlin does not support group_size={group_size}. "
            f"Supported: {MARLIN_SUPPORTED_GROUP_SIZES}"
        )


def verify_marlin_supports_shape(
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_size: int,
    group_size: int,
) -> None:
    """Verify that tensor dimensions are compatible with Marlin."""
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        raise ValueError(
            f"output_size_per_partition={output_size_per_partition} is not "
            f"divisible by min_thread_n={GPTQ_MARLIN_MIN_THREAD_N}."
        )
    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        raise ValueError(
            f"input_size_per_partition={input_size_per_partition} is not "
            f"divisible by min_thread_k={GPTQ_MARLIN_MIN_THREAD_K}."
        )
    if group_size < input_size and input_size_per_partition % group_size != 0:
        raise ValueError(
            f"input_size_per_partition={input_size_per_partition} is not "
            f"divisible by group_size={group_size}."
        )


def marlin_make_workspace(device: torch.device) -> torch.Tensor:
    """Create Marlin workspace buffer for threadblock synchronization."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms, dtype=torch.int, device=device, requires_grad=False)


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    """Create empty g_idx tensor (AWQ doesn't use activation reordering)."""
    return torch.nn.Parameter(
        torch.empty(0, dtype=torch.int, device=device), requires_grad=False
    )


def get_scale_perms():
    """Get the scale permutation indices for Marlin format."""
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]]
        )
    return scale_perm, scale_perm_single


def marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int
) -> torch.Tensor:
    """Permute quantization scales from standard to Marlin layout."""
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()
    return s


def pack_cols(
    q_w: torch.Tensor, num_bits: int, size_k: int, size_n: int
) -> torch.Tensor:
    """Pack quantized columns into int32 values."""
    pack_factor = 32 // num_bits
    assert size_n % pack_factor == 0
    out = torch.zeros(
        (size_k, size_n // pack_factor), dtype=torch.int32, device=q_w.device
    )
    for i in range(pack_factor):
        out.bitwise_or_(q_w[:, i::pack_factor].int() << (num_bits * i))
    return out


def unpack_cols(
    packed: torch.Tensor, num_bits: int, size_k: int, size_n: int
) -> torch.Tensor:
    """Unpack int32 packed columns into individual quantized values."""
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    out = torch.zeros(
        (size_k, size_n), dtype=torch.int32, device=packed.device
    )
    for i in range(pack_factor):
        out[:, i::pack_factor] = (packed >> (num_bits * i)) & mask
    return out


def marlin_zero_points(
    zp: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    """Permute and pack zero points into Marlin format."""
    scale_perm, _ = get_scale_perms()
    zp = zp.reshape((-1, len(scale_perm)))[:, scale_perm]

    # Interleave column dim (for the dequantize code) and pack to int32
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")

    zp = zp.reshape((-1, len(interleave)))[:, interleave].ravel()
    zp = zp.reshape((-1, size_n)).contiguous()
    zp = pack_cols(zp, num_bits, size_k, size_n)
    return zp


def awq_to_marlin_zero_points(
    q_zp_packed: torch.Tensor, size_k: int, size_n: int, num_bits: int
) -> torch.Tensor:
    """Convert AWQ-format zero points to Marlin format.

    AWQ zero-points are quantized and packed on the column dim with a specific
    interleaving. This function undoes the AWQ interleaving, then applies
    Marlin permutation and repacks.
    """
    q_zp = unpack_cols(q_zp_packed, num_bits, size_k, size_n)

    # Undo AWQ interleaving
    if num_bits == 4:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 4, 6, 1, 3, 5, 7]))
    elif num_bits == 8:
        undo_interleave = numpy.argsort(numpy.array([0, 2, 1, 3]))
    else:
        raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")

    q_zp = q_zp.reshape((-1, len(undo_interleave)))[:, undo_interleave].ravel()
    q_zp = q_zp.reshape((-1, size_n)).contiguous()

    return marlin_zero_points(q_zp, size_k, size_n, num_bits)


def replace_parameter(
    layer: torch.nn.Module, name: str, new_data: torch.Tensor
) -> None:
    """Replace a parameter on a layer with new data."""
    param = torch.nn.Parameter(new_data, requires_grad=False)
    layer.register_parameter(name, param)


# ---------------------------------------------------------------------------
# AWQMarlinLinearMethod
# ---------------------------------------------------------------------------

class AWQMarlinLinearMethod(LinearMethodBase):
    """Linear method for AWQ with Marlin kernel acceleration.

    Uses the Marlin W4A16 GEMM kernel for high-performance inference.
    Weights are repacked from AWQ format to Marlin format after loading.
    """

    def __init__(self, quant_config: AWQMarlinConfig) -> None:
        self.quant_config = quant_config

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
        del output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        verify_marlin_supports_shape(
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            input_size=input_size,
            group_size=group_size,
        )

        # Packed quantized weights: (input_size, output_size // pack_factor)
        qweight = Parameter(
            torch.empty(
                input_size_per_partition,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(qweight, {
            "input_dim": 0,
            "output_dim": 1,
        })
        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)

        num_groups = input_size_per_partition // group_size

        # Zero points: (num_groups, output_size // pack_factor)
        qzeros = Parameter(
            torch.empty(
                num_groups,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(qzeros, {
            "input_dim": 0,
            "output_dim": 1,
        })
        layer.register_parameter("qzeros", qzeros)
        set_weight_attrs(qzeros, extra_weight_attrs)

        # Scales: (num_groups, output_size)
        scales = Parameter(
            torch.empty(
                num_groups,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(scales, {
            "input_dim": 0,
            "output_dim": 1,
        })
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, extra_weight_attrs)

        # Store dimensions for post-loading processing
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.num_groups = num_groups

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Repack AWQ weights to Marlin format after checkpoint loading."""
        from mllm_kernel.cuda.jit.awq_marlin_repack import awq_marlin_repack

        device = layer.qweight.device

        # Unwrap parameter data for processing
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = Parameter(layer.qzeros.data, requires_grad=False)
        layer.scales = Parameter(layer.scales.data, requires_grad=False)

        # Allocate marlin workspace
        layer.workspace = marlin_make_workspace(device)

        # Repack weights from AWQ format to Marlin format
        marlin_qweight = awq_marlin_repack(
            layer.qweight,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_config.quant_type.size_bits,
        )
        replace_parameter(layer, "qweight", marlin_qweight)

        # Permute scales from AWQ format to Marlin format
        marlin_scales = marlin_permute_scales(
            layer.scales,
            size_k=layer.input_size_per_partition,
            size_n=layer.output_size_per_partition,
            group_size=self.quant_config.group_size,
        )
        replace_parameter(layer, "scales", marlin_scales)

        # Convert zero points from AWQ format to Marlin format
        marlin_zp = awq_to_marlin_zero_points(
            layer.qzeros,
            size_k=layer.num_groups,
            size_n=layer.output_size_per_partition,
            num_bits=self.quant_config.quant_type.size_bits,
        )
        replace_parameter(layer, "qzeros", marlin_zp)

        # AWQ doesn't use activation reordering
        layer.g_idx = marlin_make_empty_g_idx(device)
        layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform quantized matmul using the Marlin GEMM kernel."""
        from mllm_kernel.cuda.jit.gptq_marlin import gptq_marlin_gemm

        reshaped_x = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (layer.output_size_per_partition,)

        size_m = reshaped_x.shape[0]
        size_n = layer.output_size_per_partition
        size_k = layer.input_size_per_partition

        output = gptq_marlin_gemm(
            a=reshaped_x,
            c=None,
            b_q_weight=layer.qweight,
            b_scales=layer.scales,
            global_scale=None,
            b_zeros=layer.qzeros,
            g_idx=layer.g_idx,
            perm=layer.g_idx_sort_indices,
            workspace=layer.workspace,
            b_q_type_id=self.quant_config.quant_type.id,
            size_m=size_m,
            size_n=size_n,
            size_k=size_k,
            is_k_full=True,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

        if bias is not None:
            output.add_(bias)

        return output.reshape(out_shape)


# ---------------------------------------------------------------------------
# AWQMarlinConfig
# ---------------------------------------------------------------------------

@register_quantization("awq_marlin")
class AWQMarlinConfig(QuantizationConfig):
    """Configuration for AWQ quantization with Marlin kernel acceleration.

    This config is used when loading models quantized with AutoAWQ and
    running inference with the high-performance Marlin W4A16 GEMM kernel.

    Registered as ``"awq_marlin"`` in the quantization registry.
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        zero_point: bool,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.pack_factor = 32 // weight_bits

        if weight_bits not in _TYPE_MAP:
            raise ValueError(
                f"Unsupported weight_bits={weight_bits}. "
                f"Supported: {list(_TYPE_MAP.keys())}"
            )
        self.quant_type = _TYPE_MAP[weight_bits]

        verify_marlin_supported(
            self.quant_type,
            group_size=self.group_size,
            has_zp=self.zero_point,
        )

    def __repr__(self) -> str:
        return (
            f"AWQMarlinConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"zero_point={self.zero_point})"
        )

    def get_name(self) -> str:
        return "awq_marlin"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> AWQMarlinConfig:
        weight_bits = config.get("bits", config.get("w_bit", 4))
        group_size = config.get("group_size", 128)
        zero_point = config.get("zero_point", True)
        return cls(weight_bits, group_size, zero_point)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str = "",
    ) -> Optional[AWQMarlinLinearMethod]:
        return AWQMarlinLinearMethod(self)
