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


@tvm_ffi.register_object("mllm.DType")
class DType(tvm_ffi.Object):
    def __init__(self):
        super().__init__()


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

    def shape(self) -> tvm_ffi.Shape:
        return tvm_ffi.get_global_func("mllm.Tensor.shape")(self)


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


def is_torch_available() -> bool:
    return MLLM_FIND_TORCH_AVAILABLE is not None


def is_numpy_available() -> bool:
    return MLLM_FIND_NUMPY_AVAILABLE is not None


def from_torch(torch_tensor):
    return _ffi_api.from_torch(torch_tensor)


def from_numpy(numpy_tensor):
    return _ffi_api.from_numpy(numpy_tensor)


# Initialize context
initialize_context()


def _cleanup():
    shutdown_context()


atexit.register(_cleanup)
