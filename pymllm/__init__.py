from __future__ import annotations

from ._C import *
from . import utils
from . import nn
from . import quantize
from . import convertor

float32 = DataTypes.Float32
float16 = DataTypes.Float16
int32 = DataTypes.Int32
int16 = DataTypes.Int16
int8 = DataTypes.Int8
uint8 = DataTypes.UInt8
bfloat16 = DataTypes.BFloat16

cpu = DeviceTypes.CPU
cuda = DeviceTypes.CUDA
opencl = DeviceTypes.OpenCL

initialize_context()


def ones(shape, dtype=float32, device_type=cpu):
    return Tensor.ones(shape, dtype, device_type)


def zeros(shape, dtype=float32, device_type=cpu):
    return Tensor.zeros(shape, dtype, device_type)


def random(shape, start=-1.0, end=1.0, dtype=float32, device_type=cpu):
    return Tensor.random(shape, start, end, dtype, device_type)
