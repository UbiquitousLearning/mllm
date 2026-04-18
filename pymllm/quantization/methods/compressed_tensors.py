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


def _input_activations_cfg(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return config["config_groups"]["group_0"].get("input_activations")


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


def _per_token_quant_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token INT8 quantization using Triton kernel."""
    from pymllm.quantization.kernels.int8_activation_triton import (
        per_token_quant_int8,
    )

    return per_token_quant_int8(x)


def _int8_scaled_mm(
    x_q: torch.Tensor,
    w_q_t: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """INT8 scaled matmul using CUTLASS kernel.

    Computes: out = (x_q @ w_q_t) * x_scale * w_scale + bias
    Uses CUTLASS with per-row/col scaling epilogue fused into the GEMM.
    """
    from mllm_kernel.cuda.jit.int8_scaled_mm_cutlass import (
        int8_scaled_mm as cutlass_int8_scaled_mm,
    )

    return cutlass_int8_scaled_mm(
        x_q, w_q_t, x_scale, w_scale, out_dtype=out_dtype, bias=bias,
    )


def _validate_supported_signature(config: "CompressedTensorsConfig") -> str:
    if config.quant_format == "pack-quantized":
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
        return "w4a16"

    if config.quant_format == "int-quantized":
        if config.weight_bits != 8:
            raise ValueError(
                f"Unsupported compressed-tensors num_bits: {config.weight_bits}"
            )
        if config.group_size is not None:
            raise ValueError(
                f"Unsupported compressed-tensors group_size: {config.group_size}"
            )
        if config.weight_strategy != "channel":
            raise ValueError(
                f"Unsupported compressed-tensors weight strategy: "
                f"{config.weight_strategy}"
            )
        if config.weight_type != "int":
            raise ValueError(
                f"Unsupported compressed-tensors weight type: {config.weight_type}"
            )
        if config.weight_dynamic:
            raise ValueError("compressed-tensors int8 weights must be static")
        if not config.symmetric:
            raise ValueError("v1 only supports symmetric compressed-tensors")
        if config.actorder is not None:
            raise ValueError(
                f"Unsupported compressed-tensors actorder: {config.actorder}"
            )
        if config.input_bits != 8:
            raise ValueError(
                f"Unsupported compressed-tensors input num_bits: {config.input_bits}"
            )
        if config.input_strategy != "token":
            raise ValueError(
                f"Unsupported compressed-tensors input strategy: "
                f"{config.input_strategy}"
            )
        if config.input_type != "int":
            raise ValueError(
                f"Unsupported compressed-tensors input type: {config.input_type}"
            )
        if not config.input_dynamic:
            raise ValueError("compressed-tensors int8 inputs must be dynamic")
        if not config.input_symmetric:
            raise ValueError("v1 only supports symmetric compressed-tensors input")
        return "w8a8"

    raise ValueError(
        f"Unsupported compressed-tensors format: {config.quant_format}"
    )


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


class CompressedTensorsW8A8Int8Scheme:
    def __init__(self, *, weight_bits: int) -> None:
        self.weight_bits = weight_bits

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
        del params_dtype

        output_size_per_partition = sum(output_partition_sizes)

        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight, {"input_dim": 1, "output_dim": 0, **extra_weight_attrs}
        )
        layer.register_parameter("weight", weight)

        weight_scale = Parameter(
            torch.empty(
                output_size_per_partition,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight_scale, {"output_dim": 0, **extra_weight_attrs})
        layer.register_parameter("weight_scale", weight_scale)

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        del input_size

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.weight.dtype != torch.int8:
            raise ValueError(
                f"compressed-tensors int8 expects weight dtype int8, got "
                f"{layer.weight.dtype}"
            )

        # Store weight as (K, N) column-major for CUTLASS: stride(0)==1.
        # Original weight is (N, K) row-major. .contiguous() ensures owned memory,
        # .t() gives (K, N) with strides (1, K) = column-major.
        replace_parameter(layer, "weight", layer.weight.data.contiguous().t())

        scales = layer.weight_scale.data
        if scales.dim() == 2 and scales.shape[1] == 1:
            scales = scales[:, 0]
        elif scales.dim() != 1:
            raise ValueError(
                "compressed-tensors int8 expects weight_scale shape [N,1] or [N]"
            )
        replace_parameter(layer, "weight_scale", scales.to(torch.float32).contiguous())

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1]).contiguous()
        out_shape = x.shape[:-1] + (layer.output_size_per_partition,)

        x_q, x_scale = _per_token_quant_int8(reshaped_x)
        output = _int8_scaled_mm(
            x_q,
            layer.weight,
            x_scale,
            layer.weight_scale,
            out_dtype=x.dtype,
            bias=bias,
        )
        return output.reshape(out_shape)

class CompressedTensorsLinearMethod(LinearMethodBase):
    def __init__(
        self,
        quant_config: "CompressedTensorsConfig",
        signature: str,
    ) -> None:
        self.quant_config = quant_config
        if signature == "w4a16":
            self.scheme = CompressedTensorsWNA16Scheme(
                weight_bits=quant_config.weight_bits,
                group_size=quant_config.group_size,
                symmetric=quant_config.symmetric,
                actorder=quant_config.actorder,
            )
            return
        self.scheme = CompressedTensorsW8A8Int8Scheme(
            weight_bits=quant_config.weight_bits
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
        group_size: Optional[int],
        weight_strategy: Optional[str],
        weight_type: Optional[str],
        weight_dynamic: bool,
        symmetric: bool,
        actorder: Optional[str],
        input_bits: Optional[int],
        input_strategy: Optional[str],
        input_type: Optional[str],
        input_dynamic: bool,
        input_symmetric: bool,
    ) -> None:
        super().__init__()
        self.quant_format = quant_format
        self.ignore = ignore
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.weight_strategy = weight_strategy
        self.weight_type = weight_type
        self.weight_dynamic = weight_dynamic
        self.symmetric = symmetric
        self.actorder = actorder
        self.input_bits = input_bits
        self.input_strategy = input_strategy
        self.input_type = input_type
        self.input_dynamic = input_dynamic
        self.input_symmetric = input_symmetric

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
        input_activations = _input_activations_cfg(config)
        return cls(
            quant_format=config["format"],
            ignore=list(config.get("ignore", [])),
            weight_bits=weights["num_bits"],
            group_size=weights["group_size"],
            weight_strategy=weights.get("strategy"),
            weight_type=weights.get("type"),
            weight_dynamic=bool(weights.get("dynamic", False)),
            symmetric=weights["symmetric"],
            actorder=weights.get("actorder"),
            input_bits=(
                input_activations.get("num_bits")
                if input_activations is not None
                else None
            ),
            input_strategy=(
                input_activations.get("strategy")
                if input_activations is not None
                else None
            ),
            input_type=(
                input_activations.get("type")
                if input_activations is not None
                else None
            ),
            input_dynamic=bool(
                input_activations.get("dynamic", False)
                if input_activations is not None
                else False
            ),
            input_symmetric=bool(
                input_activations.get("symmetric", False)
                if input_activations is not None
                else False
            ),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str = ""
    ) -> Optional[CompressedTensorsLinearMethod]:
        signature = _validate_supported_signature(self)
        if any(ignored and prefix.startswith(ignored) for ignored in self.ignore):
            return None
        return CompressedTensorsLinearMethod(self, signature)
