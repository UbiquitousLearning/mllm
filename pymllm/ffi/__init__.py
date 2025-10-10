# Copyright (c) MLLM Team.
# Licensed under the MIT License.

from __future__ import annotations
import tvm_ffi
import atexit
from .base import _LIB
from . import _ffi_api
from typing import Union

import importlib.util

MLLM_FIND_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
MLLM_FIND_NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


def echo(rec: str) -> None:
    return _ffi_api.echo(rec)


def initialize_context() -> None:
    return _ffi_api.initialize_context()


def shutdown_context() -> None:
    return _ffi_api.shutdown_context()


@tvm_ffi.register_object("mllm.Device")
class Device(tvm_ffi.Object):
    def __init__(self):
        super().__init__()

    def to_pod(self) -> int:
        return tvm_ffi.get_global_func("mllm.Device.to_pod")(self)


@tvm_ffi.register_object("mllm.DType")
class DType(tvm_ffi.Object):
    def __init__(self):
        super().__init__()

    def to_pod(self) -> int:
        return tvm_ffi.get_global_func("mllm.DType.to_pod")(self)


def float32_() -> DType:
    return _ffi_api.float32_()


def float16_() -> DType:
    return _ffi_api.float16_()


def bfloat16_() -> DType:
    return _ffi_api.bfloat16_()


def cpu_() -> Device:
    return _ffi_api.cpu_()


def cuda_() -> Device:
    return _ffi_api.float32_()


def qnn_() -> Device:
    return _ffi_api.qnn_()


@tvm_ffi.register_object("mllm.Tensor")
class Tensor(tvm_ffi.Object):
    def __init__(self):
        self.__init_handle_by_constructor__(Tensor.__create__)

    def __str__(self) -> str:
        return tvm_ffi.get_global_func("mllm.Tensor.str")(self)

    @property
    def shape(self) -> tvm_ffi.Shape:
        return tvm_ffi.get_global_func("mllm.Tensor.shape")(self)

    @property
    def dtype(self) -> DType:
        return tvm_ffi.get_global_func("mllm.Tensor.dtype")(self)

    @property
    def device(self) -> Device:
        return tvm_ffi.get_global_func("mllm.Tensor.device")(self)

    def tobytes(self) -> tvm_ffi.Array:
        tvm_bytes: tvm_ffi.Array = tvm_ffi.get_global_func("mllm.Tensor.tobytes")(self)
        return tvm_bytes

    def __add__(self, other) -> Tensor:
        if isinstance(other, Tensor):
            return tvm_ffi.get_global_func("mllm.Tensor.add")(self, other)
        elif isinstance(other, (int, float)):
            return tvm_ffi.get_global_func("mllm.Tensor.add_scalar")(self, other)
        else:
            raise TypeError(
                "Addition is not supported between Tensor and {}".format(type(other))
            )

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return tvm_ffi.get_global_func("mllm.Tensor.sub")(self, other)
        elif isinstance(other, (int, float)):
            return tvm_ffi.get_global_func("mllm.Tensor.sub_scalar")(self, other)
        else:
            raise TypeError(
                "Subtraction is not supported between Tensor and {}".format(type(other))
            )

    def __mul__(self, other) -> Tensor:
        if isinstance(other, Tensor):
            return tvm_ffi.get_global_func("mllm.Tensor.mul")(self, other)
        elif isinstance(other, (int, float)):
            return tvm_ffi.get_global_func("mllm.Tensor.mul_scalar")(self, other)
        else:
            raise TypeError(
                "Multiplication is not supported between Tensor and {}".format(
                    type(other)
                )
            )

    def __div__(self, other) -> Tensor:
        if isinstance(other, Tensor):
            return tvm_ffi.get_global_func("mllm.Tensor.div")(self, other)
        elif isinstance(other, (int, float)):
            return tvm_ffi.get_global_func("mllm.Tensor.div_scalar")(self, other)
        else:
            raise TypeError(
                "Division is not supported between Tensor and {}".format(type(other))
            )

    def __neg__(self, other) -> Tensor:
        return tvm_ffi.get_global_func("mllm.Tensor.neg")(self, other)

    def abs(self) -> Tensor:
        return tvm_ffi.get_global_func("mllm.Tensor.abs")(self)

    def clip(self, min_val: float, max_val: float) -> Tensor:
        return tvm_ffi.get_global_func("mllm.Tensor.clip")(self, min_val, max_val)

    def min(self, dim: int = -1, keep_dim: bool = False) -> Tensor:
        return tvm_ffi.get_global_func("mllm.Tensor.min")(self, dim, keep_dim)

    def max(self, dim: int = -1, keep_dim: bool = False) -> Tensor:
        return tvm_ffi.get_global_func("mllm.Tensor.max")(self, dim, keep_dim)

    def sum(self, dim: int = -1, keep_dim: bool = False) -> Tensor:
        return tvm_ffi.get_global_func("mllm.Tensor.sum")(self, dim, keep_dim)

    def mean(self, dim: int = -1, keep_dim: bool = False) -> Tensor:
        return tvm_ffi.get_global_func("mllm.Tensor.mean")(self, dim, keep_dim)

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        return tvm_ffi.get_global_func("mllm.Tensor.transpose")(self, dim0, dim1)

    @property
    def T(self) -> Tensor:
        return tvm_ffi.get_global_func("mllm.Tensor.T")(self)

    def view(self, shape) -> Tensor:
        return tvm_ffi.get_global_func("mllm.Tensor.view")(self, shape)

    def unsqueeze(self, dim: int) -> Tensor:
        return tvm_ffi.get_global_func("mllm.Tensor.unsqueeze")(self, dim)

    def squeeze(self, dim: int) -> Tensor:
        return tvm_ffi.get_global_func("mllm.Tensor.squeeze")(self, dim)

    def permute(self, dims) -> Tensor:
        return tvm_ffi.get_global_func("mllm.Tensor.permute")(self, dims)

    def contiguous(self) -> Tensor:
        return tvm_ffi.get_global_func("mllm.Tensor.contiguous")(self)

    def clone(self) -> Tensor:
        return tvm_ffi.get_global_func("mllm.Tensor.clone")(self)

    def repeat(self, multiplier, dim) -> Tensor:
        return tvm_ffi.get_global_func("mllm.Tensor.repeat")(self, multiplier, dim)

    def to(self, dd: Union[Device, DType]) -> Tensor:
        if isinstance(dd, DType):
            return tvm_ffi.get_global_func("mllm.Tensor.to_dtype")(self, dd)
        elif isinstance(dd, Device):
            return tvm_ffi.get_global_func("mllm.Tensor.to_device")(self, dd)
        else:
            raise ValueError("Invalid device or dtype")

    def cpu(self) -> Tensor:
        return tvm_ffi.get_global_func("mllm.Tensor.cpu")(self)

    def cuda(self) -> Tensor:
        return tvm_ffi.get_global_func("mllm.Tensor.cuda")(self)

    @property
    def name(self):
        return tvm_ffi.get_global_func("mllm.Tensor.get_name")(self)

    def set_name(self, name):
        tvm_ffi.get_global_func("mllm.Tensor.set_name")(self, name)

    def numel(self):
        return tvm_ffi.get_global_func("mllm.Tensor.numel")(self)

    @property
    def rank(self):
        return tvm_ffi.get_global_func("mllm.Tensor.rank")(self)

    def is_contiguous(self):
        return tvm_ffi.get_global_func("mllm.Tensor.is_contiguous")(self)


# Global dtypes
float32: DType = float32_()
float16: DType = float16_()
bfloat16: DType = bfloat16_()
cpu: Device = cpu_()
cuda: Device = cuda_()
qnn: Device = qnn_()


def device(device_type: str) -> Device:
    if device_type == "cpu":
        return cpu
    elif device_type == "cuda":
        return cuda
    elif device_type == "qnn":
        return qnn
    else:
        raise ValueError("Invalid device type: {}".format(device_type))


def empty(
    shape: tvm_ffi.Shape, dtype: DType = float32, device_type: Union[Device, str] = cpu
) -> Tensor:
    if isinstance(device_type, str):
        device_type = device(device_type)
    return _ffi_api.empty(shape, dtype, device_type)


def zeros(
    shape: tvm_ffi.Shape, dtype: DType = float32, device_type: Union[Device, str] = cpu
) -> Tensor:
    if isinstance(device_type, str):
        device_type = device(device_type)
    return _ffi_api.zeros(shape, dtype, device_type)


def ones(
    shape: tvm_ffi.Shape, dtype: DType = float32, device_type: Union[Device, str] = cpu
) -> Tensor:
    if isinstance(device_type, str):
        device_type = device(device_type)
    return _ffi_api.ones(shape, dtype, device_type)


def arange(
    start: float,
    end: float,
    step: float = 1,
    dtype: DType = float32,
    device_type: Union[Device, str] = cpu,
) -> Tensor:
    if isinstance(device_type, str):
        device_type = device(device_type)
    return _ffi_api.arange(start, end, step, dtype, device_type)


def random(
    shape: tvm_ffi.Shape,
    start: float = -1.0,
    end: float = 1.0,
    dtype: DType = float32,
    device_type: Union[Device, str] = cpu,
) -> Tensor:
    if isinstance(device_type, str):
        device_type = device(device_type)
    return _ffi_api.random(shape, start, end, dtype, device_type)


def is_torch_available() -> bool:
    return MLLM_FIND_TORCH_AVAILABLE is not None


def is_numpy_available() -> bool:
    return MLLM_FIND_NUMPY_AVAILABLE is not None


def from_torch(torch_tensor):
    return _ffi_api.from_torch(torch_tensor)


def from_numpy(numpy_tensor):
    return _ffi_api.from_numpy(numpy_tensor)


@tvm_ffi.register_object("mllm.service.Session")
class Session(tvm_ffi.Object):
    def __init__(self):
        pass


# Initialize context
initialize_context()


def _cleanup():
    shutdown_context()


atexit.register(_cleanup)
