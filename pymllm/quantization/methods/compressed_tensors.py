from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.nn import Parameter

from mllm_kernel.cuda.jit import gptq_marlin_gemm, gptq_marlin_repack
from pymllm.layers.quantize_base import LinearMethodBase
from pymllm.layers.utils import set_weight_attrs
from pymllm.quantization.quant_config import QuantizationConfig, register_quantization

MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_TILE = 16


class _ScalarTypeInfo:
    def __init__(self, name: str, size_bits: int, type_id: int):
        self.name = name
        self.size_bits = size_bits
        self.id = type_id


def _compute_scalar_type_id(
    exponent: int,
    mantissa: int,
    signed: bool,
    bias: int,
    finite_values_only: bool = False,
    nan_repr: int = 1,
) -> int:
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
        result |= (int(value) & ((1 << width) - 1)) << bit_offset
        bit_offset += width
    return result


SCALAR_TYPE_UINT4 = _ScalarTypeInfo(
    "uint4", 4, _compute_scalar_type_id(0, 4, False, 0)
)
SCALAR_TYPE_UINT4B8 = _ScalarTypeInfo(
    "uint4b8", 4, _compute_scalar_type_id(0, 4, False, 8)
)


def _weights_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return config["config_groups"]["group_0"]["weights"]


def verify_marlin_supported(group_size: int) -> None:
    if group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
        raise ValueError(
            f"Unsupported compressed-tensors group_size: {group_size}"
        )
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 80:
        raise ValueError("compressed-tensors Marlin requires SM80+")


def verify_marlin_supports_shape(
    output_size_per_partition: int,
    input_size_per_partition: int,
    input_size: int,
    group_size: int,
) -> None:
    if output_size_per_partition % GPTQ_MARLIN_MIN_THREAD_N != 0:
        raise ValueError("output_size_per_partition must be divisible by 64")
    if input_size_per_partition % GPTQ_MARLIN_MIN_THREAD_K != 0:
        raise ValueError("input_size_per_partition must be divisible by 128")
    if group_size < input_size and input_size_per_partition % group_size != 0:
        raise ValueError(
            "input_size_per_partition must be divisible by group_size"
        )


def marlin_make_workspace(device: torch.device) -> torch.Tensor:
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms, dtype=torch.int, device=device, requires_grad=False)


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return Parameter(
        torch.empty(0, dtype=torch.int32, device=device), requires_grad=False
    )


def get_scale_perms():
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
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


def replace_parameter(
    layer: torch.nn.Module, name: str, new_data: torch.Tensor
) -> None:
    layer.register_parameter(name, Parameter(new_data, requires_grad=False))


def _validate_supported_signature(config: "CompressedTensorsConfig") -> None:
    if config.quant_format != "pack-quantized":
        raise ValueError(
            f"Unsupported compressed-tensors format: {config.quant_format}"
        )
    if config.weight_bits != 4:
        raise ValueError(
            f"Unsupported compressed-tensors num_bits: {config.weight_bits}"
        )
    if config.group_size != 32:
        raise ValueError(
            f"Unsupported compressed-tensors group_size: {config.group_size}"
        )
    if not config.symmetric:
        raise ValueError("v1 only supports symmetric compressed-tensors")
    if config.actorder is not None:
        raise ValueError(
            f"Unsupported compressed-tensors actorder: {config.actorder}"
        )
    verify_marlin_supported(config.group_size)


class CompressedTensorsWNA16Scheme:
    def __init__(
        self,
        *,
        weight_bits: int,
        group_size: int,
        symmetric: bool,
        actorder: Optional[str],
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.actorder = actorder
        self.pack_factor = 32 // weight_bits
        self.quant_type = (
            SCALAR_TYPE_UINT4B8 if symmetric else SCALAR_TYPE_UINT4
        )

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
        verify_marlin_supports_shape(
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            input_size=input_size,
            group_size=self.group_size,
        )

        weight_packed = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight_packed, {"output_dim": 0, **extra_weight_attrs})
        layer.register_parameter("weight_packed", weight_packed)

        weight_scale = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight_scale, {"output_dim": 0, **extra_weight_attrs})
        layer.register_parameter("weight_scale", weight_scale)

        weight_shape = Parameter(torch.empty(2, dtype=torch.int64), requires_grad=False)
        set_weight_attrs(weight_shape, extra_weight_attrs)
        layer.register_parameter("weight_shape", weight_shape)

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.group_size = self.group_size

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.weight_packed.device
        size_k = layer.input_size_per_partition
        size_n = layer.output_size_per_partition

        verify_marlin_supports_shape(
            output_size_per_partition=size_n,
            input_size_per_partition=size_k,
            input_size=size_k,
            group_size=self.group_size,
        )

        layer.workspace = marlin_make_workspace(device)
        layer.weight_zero_point = marlin_make_empty_g_idx(device)
        layer.weight_g_idx = marlin_make_empty_g_idx(device)
        layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        repacked_qweight = gptq_marlin_repack(
            layer.weight_packed.data.t().contiguous(),
            perm=layer.g_idx_sort_indices,
            size_k=size_k,
            size_n=size_n,
            num_bits=self.weight_bits,
        )
        repacked_scales = marlin_permute_scales(
            layer.weight_scale.data.t().contiguous(),
            size_k=size_k,
            size_n=size_n,
            group_size=self.group_size,
        )

        replace_parameter(layer, "weight_packed", repacked_qweight)
        replace_parameter(layer, "weight_scale", repacked_scales)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (layer.output_size_per_partition,)
        output = gptq_marlin_gemm(
            a=reshaped_x,
            c=None,
            b_q_weight=layer.weight_packed,
            b_scales=layer.weight_scale,
            global_scale=None,
            b_zeros=layer.weight_zero_point,
            g_idx=layer.weight_g_idx,
            perm=layer.g_idx_sort_indices,
            workspace=layer.workspace,
            b_q_type_id=self.quant_type.id,
            size_m=reshaped_x.shape[0],
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            is_k_full=True,
            use_fp32_reduce=True,
            is_zp_float=False,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)


class CompressedTensorsLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: "CompressedTensorsConfig") -> None:
        self.quant_config = quant_config
        self.scheme = CompressedTensorsWNA16Scheme(
            weight_bits=quant_config.weight_bits,
            group_size=quant_config.group_size,
            symmetric=quant_config.symmetric,
            actorder=quant_config.actorder,
        )

    def create_weights(self, *args: Any, **kwargs: Any) -> None:
        self.scheme.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.scheme.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.scheme.apply(layer, x, bias)


@register_quantization("compressed-tensors")
class CompressedTensorsConfig(QuantizationConfig):
    def __init__(
        self,
        *,
        quant_format: str,
        ignore: List[str],
        weight_bits: int,
        group_size: int,
        symmetric: bool,
        actorder: Optional[str],
    ) -> None:
        super().__init__()
        self.quant_format = quant_format
        self.ignore = ignore
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.actorder = actorder

    def get_name(self) -> str:
        return "compressed-tensors"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return ["config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CompressedTensorsConfig":
        weights = _weights_cfg(config)
        return cls(
            quant_format=config["format"],
            ignore=list(config.get("ignore", [])),
            weight_bits=weights["num_bits"],
            group_size=weights["group_size"],
            symmetric=weights["symmetric"],
            actorder=weights.get("actorder"),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str = ""
    ) -> Optional[CompressedTensorsLinearMethod]:
        _validate_supported_signature(self)
        if any(ignored and prefix.startswith(ignored) for ignored in self.ignore):
            return None
        return CompressedTensorsLinearMethod(self)
