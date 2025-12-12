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


@tvm_ffi.register_object("mllm.ParameterFile")
class ParameterFile(tvm_ffi.Object):
    def __init__(self):
        pass


@tvm_ffi.register_object("mllm.BaseOp")
class BaseOp(tvm_ffi.Object):
    def __init__(self):
        pass

    def load(self, pf: ParameterFile):
        return tvm_ffi.get_global_func("mllm.BaseOp.load")(self, pf)


@tvm_ffi.register_object("mllm.qualcomm.QnnDeviceAndContext")
class QnnDeviceAndContext(tvm_ffi.Object):
    def __init__(self):
        pass


@tvm_ffi.register_object("mllm.qualcomm.QcomHTPArch")
class QcomHTPArch(tvm_ffi.Object):
    def __init__(self):
        pass

    @staticmethod
    def NONE() -> QcomHTPArch:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomHTPArch.NONE")()

    @staticmethod
    def V68() -> QcomHTPArch:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomHTPArch.V68")()

    @staticmethod
    def V69() -> QcomHTPArch:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomHTPArch.V69")()

    @staticmethod
    def V73() -> QcomHTPArch:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomHTPArch.V73")()

    @staticmethod
    def V75() -> QcomHTPArch:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomHTPArch.V75")()

    @staticmethod
    def V79() -> QcomHTPArch:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomHTPArch.V79")()

    @staticmethod
    def V81() -> QcomHTPArch:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomHTPArch.V81")()


@tvm_ffi.register_object("mllm.qualcomm.QcomChipset")
class QcomChipset(tvm_ffi.Object):
    def __init__(self):
        pass

    @staticmethod
    def UNKNOWN_SM() -> QcomChipset:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomChipset.UNKNOWN_SM")()

    @staticmethod
    def SA8295() -> QcomChipset:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomChipset.SA8295")()

    @staticmethod
    def SM8350() -> QcomChipset:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomChipset.SM8350")()

    @staticmethod
    def SM8450() -> QcomChipset:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomChipset.SM8450")()

    @staticmethod
    def SM8475() -> QcomChipset:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomChipset.SM8475")()

    @staticmethod
    def SM8550() -> QcomChipset:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomChipset.SM8550")()

    @staticmethod
    def SM8650() -> QcomChipset:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomChipset.SM8650")()

    @staticmethod
    def SM8750() -> QcomChipset:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomChipset.SM8750")()

    @staticmethod
    def SM8850() -> QcomChipset:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomChipset.SM8850")()

    @staticmethod
    def SSG2115P() -> QcomChipset:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomChipset.SSG2115P")()

    @staticmethod
    def SSG2125P() -> QcomChipset:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomChipset.SSG2125P")()

    @staticmethod
    def SXR1230P() -> QcomChipset:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomChipset.SXR1230P")()

    @staticmethod
    def SXR2230P() -> QcomChipset:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomChipset.SXR2230P")()

    @staticmethod
    def SXR2330P() -> QcomChipset:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomChipset.SXR2330P")()

    @staticmethod
    def QCS9100() -> QcomChipset:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomChipset.QCS9100")()

    @staticmethod
    def SAR2230P() -> QcomChipset:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomChipset.SAR2230P")()

    @staticmethod
    def SA8255() -> QcomChipset:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomChipset.SA8255")()

    @staticmethod
    def SW6100() -> QcomChipset:
        return tvm_ffi.get_global_func("mllm.qualcomm.QcomChipset.SW6100")()


@tvm_ffi.register_object("mllm.qualcomm.QcomTryBestPerformance")
class QcomTryBestPerformance(tvm_ffi.Object):
    def __init__(self):
        pass

    @staticmethod
    def HtpDefault() -> QcomTryBestPerformance:
        return tvm_ffi.get_global_func(
            "mllm.qualcomm.QcomTryBestPerformance.HtpDefault"
        )()

    @staticmethod
    def HtpSustainedHighPerformance() -> QcomTryBestPerformance:
        return tvm_ffi.get_global_func(
            "mllm.qualcomm.QcomTryBestPerformance.HtpSustainedHighPerformance"
        )()

    @staticmethod
    def HtpBurst() -> QcomTryBestPerformance:
        return tvm_ffi.get_global_func(
            "mllm.qualcomm.QcomTryBestPerformance.HtpBurst"
        )()

    @staticmethod
    def HtpHighPerformance() -> QcomTryBestPerformance:
        return tvm_ffi.get_global_func(
            "mllm.qualcomm.QcomTryBestPerformance.HtpHighPerformance"
        )()

    @staticmethod
    def HtpPowerSaver() -> QcomTryBestPerformance:
        return tvm_ffi.get_global_func(
            "mllm.qualcomm.QcomTryBestPerformance.HtpPowerSaver"
        )()

    @staticmethod
    def HtpLowPowerSaver() -> QcomTryBestPerformance:
        return tvm_ffi.get_global_func(
            "mllm.qualcomm.QcomTryBestPerformance.HtpLowPowerSaver"
        )()

    @staticmethod
    def HtpHighPowerSaver() -> QcomTryBestPerformance:
        return tvm_ffi.get_global_func(
            "mllm.qualcomm.QcomTryBestPerformance.HtpHighPowerSaver"
        )()

    @staticmethod
    def HtpLowBalanced() -> QcomTryBestPerformance:
        return tvm_ffi.get_global_func(
            "mllm.qualcomm.QcomTryBestPerformance.HtpLowBalanced"
        )()

    @staticmethod
    def HtpBalanced() -> QcomTryBestPerformance:
        return tvm_ffi.get_global_func(
            "mllm.qualcomm.QcomTryBestPerformance.HtpBalanced"
        )()


@tvm_ffi.register_object("mllm.qualcomm.QcomSecurityPDSession")
class QcomSecurityPDSession(tvm_ffi.Object):
    def __init__(self):
        pass

    @staticmethod
    def HtpUnsignedPd() -> QcomSecurityPDSession:
        return tvm_ffi.get_global_func(
            "mllm.qualcomm.QcomSecurityPDSession.HtpUnsignedPd"
        )()

    @staticmethod
    def HtpSignedPd() -> QcomSecurityPDSession:
        return tvm_ffi.get_global_func(
            "mllm.qualcomm.QcomSecurityPDSession.HtpSignedPd"
        )()


@tvm_ffi.register_object("mllm.qualcomm.QcomTargetMachine")
class QcomTargetMachine(tvm_ffi.Object):
    def __init__(
        self,
        soc_htp_chipset: QcomChipset,
        soc_htp_arch: QcomHTPArch,
        soc_htp_performance: QcomTryBestPerformance,
        soc_htp_security_pd_session: QcomSecurityPDSession,
        soc_htp_vtcm: int,
    ):
        self.__init_handle_by_constructor__(
            QcomTargetMachine.__create__,
            soc_htp_chipset,
            soc_htp_arch,
            soc_htp_performance,
            soc_htp_security_pd_session,
            soc_htp_vtcm,
        )


@tvm_ffi.register_object("mllm.qualcomm.QnnAOTEnv")
class QnnAOTEnv(tvm_ffi.Object):
    def __init__(
        self,
        machine: QcomTargetMachine = None,
        path: str = None,
    ):
        if machine is None:
            raise RuntimeError("machine target is none!")
        if path is None or path == "":
            self.__init_handle_by_constructor__(QnnAOTEnv.__create__, machine, "")
        else:
            self.__init_handle_by_constructor__(QnnAOTEnv.__create__, machine, path)

    def create_context(
        self, name: str, weights_sharing: bool = False
    ) -> QnnDeviceAndContext:
        return tvm_ffi.get_global_func("mllm.qualcomm.QnnAOTEnv.createContext")(
            self, name, weights_sharing
        )


# =============================================================================
# Mllm Ops Binding
#
# =============================================================================
@tvm_ffi.register_object("mllm.aops.SoftmaxOpOptions")
class SoftmaxOpOptions(tvm_ffi.Object):
    def __init__(self, dim=-1):
        super().__init__()
        self.__init_handle_by_constructor__(SoftmaxOpOptions.__create__, dim)


@tvm_ffi.register_object("mllm.aops.SoftmaxOp")
class SoftmaxOp(BaseOp):
    def __init__(self):
        super().__init__()

    @staticmethod
    def create(device: Device, options: SoftmaxOpOptions):
        return tvm_ffi.get_global_func("mllm.aops.__ctx_create_softmax_op")(
            device, options
        )


# Initialize context
initialize_context()


def _cleanup():
    shutdown_context()


atexit.register(_cleanup)
