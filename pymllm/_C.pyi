from __future__ import annotations
import typing
import typing_extensions
__all__ = ['AbstractNnNode', 'ConfigFile', 'Context', 'DataTypes', 'DeviceTypes', 'Dispatcher', 'DispatcherManager', 'DispatcherManagerOptions', 'LayerImpl', 'MemoryManager', 'MemoryManagerOptions', 'ModelFileVersion', 'ModuleImpl', 'PerfFile', 'SessionTCB', 'Task', 'TaskTypes', 'Tensor', 'clean_this_thread', 'get_perf_file', 'initialize_context', 'is_opencl_available', 'is_qnn_available', 'load', 'memory_report', 'perf_end', 'perf_start', 'save', 'set_maximum_num_threads', 'set_random_seed', 'shutdown_context', 'this_thread']
class AbstractNnNode:
    def depthDecrease(self) -> None:
        ...
    def depthIncrease(self) -> None:
        ...
    def getAbsoluteName(self) -> str:
        ...
    def getDepth(self) -> int:
        ...
    def getDevice(self) -> DeviceTypes:
        ...
    def getName(self) -> str:
        ...
    def getType(self) -> ...:
        ...
    def isCompiledAsObj(self) -> bool:
        ...
    def refChildNodes(self) -> list[AbstractNnNode]:
        ...
    def refParentNode(self) -> ...:
        ...
    def regChildNode(self, arg0: AbstractNnNode) -> None:
        ...
    def setAbsoluteName(self, arg0: str) -> None:
        ...
    def setCompiledAsObj(self, arg0: bool) -> None:
        ...
    def setDepth(self, arg0: typing.SupportsInt) -> None:
        ...
    def setName(self, arg0: str) -> None:
        ...
class ConfigFile:
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    def data(self) -> typing.Any:
        ...
    def dump(self) -> str:
        ...
    def load(self, arg0: str) -> None:
        ...
    def loadString(self, arg0: str) -> None:
        ...
    def save(self, arg0: str) -> None:
        ...
class Context:
    @staticmethod
    def instance() -> Context:
        ...
    def dispatcherManager(self) -> DispatcherManager:
        ...
    def getPerfFile(self) -> PerfFile:
        ...
    def getRandomSeed(self) -> int:
        ...
    def getUUID(self) -> int:
        ...
    def isPerfMode(self) -> bool:
        ...
    def mainThread(self) -> SessionTCB:
        ...
    def memoryManager(self) -> MemoryManager:
        ...
    def refSessionThreads(self) -> ...:
        ...
    def setPerfMode(self, arg0: bool) -> None:
        ...
    def setRandomSeed(self, arg0: typing.SupportsInt) -> None:
        ...
    def thisThread(self) -> SessionTCB:
        ...
class DataTypes:
    """
    Members:
    
      Float32
    
      Float16
    
      GGUF_Q4_0
    
      GGUF_Q4_1
    
      GGUF_Q8_0
    
      GGUF_Q8_1
    
      GGUF_Q8_Pertensor
    
      GGUF_Q4_K
    
      GGUF_Q6_K
    
      GGUF_Q8_K
    
      Int8
    
      Int16
    
      Int32
    
      GGUF_Q4_0_4_4
    
      GGUF_Q4_0_4_8
    
      GGUF_Q4_0_8_8
    
      GGUF_Q8_0_4_4
    
      GGUF_Q3_K
    
      GGUF_Q2_K
    
      GGUF_Q1_K
    
      GGUF_IQ2_XXS
    
      GGUF_IQ2_XS
    
      GGUF_IQ1_S
    
      GGUF_IQ1_M
    
      GGUF_IQ2_S
    
      BFloat16
    
      UInt8
    
      UInt16
    
      UInt32
    
      Int64
    
      UInt64
    
      Byte
    """
    BFloat16: typing.ClassVar[DataTypes]  # value = <DataTypes.BFloat16: 128>
    Byte: typing.ClassVar[DataTypes]  # value = <DataTypes.Byte: 134>
    Float16: typing.ClassVar[DataTypes]  # value = <DataTypes.Float16: 1>
    Float32: typing.ClassVar[DataTypes]  # value = <DataTypes.Float32: 0>
    GGUF_IQ1_M: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_IQ1_M: 29>
    GGUF_IQ1_S: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_IQ1_S: 28>
    GGUF_IQ2_S: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_IQ2_S: 30>
    GGUF_IQ2_XS: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_IQ2_XS: 27>
    GGUF_IQ2_XXS: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_IQ2_XXS: 26>
    GGUF_Q1_K: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_Q1_K: 25>
    GGUF_Q2_K: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_Q2_K: 24>
    GGUF_Q3_K: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_Q3_K: 23>
    GGUF_Q4_0: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_Q4_0: 2>
    GGUF_Q4_0_4_4: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_Q4_0_4_4: 19>
    GGUF_Q4_0_4_8: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_Q4_0_4_8: 20>
    GGUF_Q4_0_8_8: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_Q4_0_8_8: 21>
    GGUF_Q4_1: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_Q4_1: 3>
    GGUF_Q4_K: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_Q4_K: 12>
    GGUF_Q6_K: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_Q6_K: 14>
    GGUF_Q8_0: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_Q8_0: 8>
    GGUF_Q8_0_4_4: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_Q8_0_4_4: 22>
    GGUF_Q8_1: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_Q8_1: 9>
    GGUF_Q8_K: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_Q8_K: 15>
    GGUF_Q8_Pertensor: typing.ClassVar[DataTypes]  # value = <DataTypes.GGUF_Q8_Pertensor: 10>
    Int16: typing.ClassVar[DataTypes]  # value = <DataTypes.Int16: 17>
    Int32: typing.ClassVar[DataTypes]  # value = <DataTypes.Int32: 18>
    Int64: typing.ClassVar[DataTypes]  # value = <DataTypes.Int64: 132>
    Int8: typing.ClassVar[DataTypes]  # value = <DataTypes.Int8: 16>
    UInt16: typing.ClassVar[DataTypes]  # value = <DataTypes.UInt16: 130>
    UInt32: typing.ClassVar[DataTypes]  # value = <DataTypes.UInt32: 131>
    UInt64: typing.ClassVar[DataTypes]  # value = <DataTypes.UInt64: 133>
    UInt8: typing.ClassVar[DataTypes]  # value = <DataTypes.UInt8: 129>
    __members__: typing.ClassVar[dict[str, DataTypes]]  # value = {'Float32': <DataTypes.Float32: 0>, 'Float16': <DataTypes.Float16: 1>, 'GGUF_Q4_0': <DataTypes.GGUF_Q4_0: 2>, 'GGUF_Q4_1': <DataTypes.GGUF_Q4_1: 3>, 'GGUF_Q8_0': <DataTypes.GGUF_Q8_0: 8>, 'GGUF_Q8_1': <DataTypes.GGUF_Q8_1: 9>, 'GGUF_Q8_Pertensor': <DataTypes.GGUF_Q8_Pertensor: 10>, 'GGUF_Q4_K': <DataTypes.GGUF_Q4_K: 12>, 'GGUF_Q6_K': <DataTypes.GGUF_Q6_K: 14>, 'GGUF_Q8_K': <DataTypes.GGUF_Q8_K: 15>, 'Int8': <DataTypes.Int8: 16>, 'Int16': <DataTypes.Int16: 17>, 'Int32': <DataTypes.Int32: 18>, 'GGUF_Q4_0_4_4': <DataTypes.GGUF_Q4_0_4_4: 19>, 'GGUF_Q4_0_4_8': <DataTypes.GGUF_Q4_0_4_8: 20>, 'GGUF_Q4_0_8_8': <DataTypes.GGUF_Q4_0_8_8: 21>, 'GGUF_Q8_0_4_4': <DataTypes.GGUF_Q8_0_4_4: 22>, 'GGUF_Q3_K': <DataTypes.GGUF_Q3_K: 23>, 'GGUF_Q2_K': <DataTypes.GGUF_Q2_K: 24>, 'GGUF_Q1_K': <DataTypes.GGUF_Q1_K: 25>, 'GGUF_IQ2_XXS': <DataTypes.GGUF_IQ2_XXS: 26>, 'GGUF_IQ2_XS': <DataTypes.GGUF_IQ2_XS: 27>, 'GGUF_IQ1_S': <DataTypes.GGUF_IQ1_S: 28>, 'GGUF_IQ1_M': <DataTypes.GGUF_IQ1_M: 29>, 'GGUF_IQ2_S': <DataTypes.GGUF_IQ2_S: 30>, 'BFloat16': <DataTypes.BFloat16: 128>, 'UInt8': <DataTypes.UInt8: 129>, 'UInt16': <DataTypes.UInt16: 130>, 'UInt32': <DataTypes.UInt32: 131>, 'Int64': <DataTypes.Int64: 132>, 'UInt64': <DataTypes.UInt64: 133>, 'Byte': <DataTypes.Byte: 134>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class DeviceTypes:
    """
    Members:
    
      CPU
    
      CUDA
    
      OpenCL
    """
    CPU: typing.ClassVar[DeviceTypes]  # value = <DeviceTypes.CPU: 1>
    CUDA: typing.ClassVar[DeviceTypes]  # value = <DeviceTypes.CUDA: 2>
    OpenCL: typing.ClassVar[DeviceTypes]  # value = <DeviceTypes.OpenCL: 3>
    __members__: typing.ClassVar[dict[str, DeviceTypes]]  # value = {'CPU': <DeviceTypes.CPU: 1>, 'CUDA': <DeviceTypes.CUDA: 2>, 'OpenCL': <DeviceTypes.OpenCL: 3>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Dispatcher:
    def id(self) -> int:
        ...
    def process(self, arg0: Task) -> None:
        ...
    def syncWait(self) -> None:
        ...
class DispatcherManager:
    def submit(self, arg0: typing.SupportsInt, arg1: Task) -> None:
        ...
    def syncWait(self, arg0: typing.SupportsInt) -> None:
        ...
class DispatcherManagerOptions:
    numa_policy: bool
    def __init__(self) -> None:
        ...
    @property
    def num_threads(self) -> int:
        ...
    @num_threads.setter
    def num_threads(self, arg0: typing.SupportsInt) -> None:
        ...
class LayerImpl(AbstractNnNode):
    def getInstancedOp(self) -> ...:
        ...
    def load(self, arg0: ...) -> None:
        ...
    def opType(self) -> ...:
        ...
    def refOptions(self) -> ...:
        ...
    def setInstancedOp(self, arg0: ...) -> None:
        ...
    def to(self, arg0: DeviceTypes) -> None:
        ...
class MemoryManager:
    def clearAll(self) -> None:
        ...
    def report(self) -> None:
        ...
class MemoryManagerOptions:
    using_buddy_mem_pool: bool
    def __init__(self) -> None:
        ...
    @property
    def really_large_tensor_threshold(self) -> int:
        ...
    @really_large_tensor_threshold.setter
    def really_large_tensor_threshold(self, arg0: typing.SupportsInt) -> None:
        ...
class ModelFileVersion:
    """
    Members:
    
      V1
    
      V2
    
      UserTemporary
    """
    UserTemporary: typing.ClassVar[ModelFileVersion]  # value = <ModelFileVersion.UserTemporary: 0>
    V1: typing.ClassVar[ModelFileVersion]  # value = <ModelFileVersion.V1: 1>
    V2: typing.ClassVar[ModelFileVersion]  # value = <ModelFileVersion.V2: 2>
    __members__: typing.ClassVar[dict[str, ModelFileVersion]]  # value = {'V1': <ModelFileVersion.V1: 1>, 'V2': <ModelFileVersion.V2: 2>, 'UserTemporary': <ModelFileVersion.UserTemporary: 0>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ModuleImpl(AbstractNnNode):
    def __init__(self) -> None:
        ...
    def load(self, arg0: ...) -> None:
        ...
    def params(self, arg0: ModelFileVersion) -> ...:
        ...
    def to(self, arg0: DeviceTypes) -> None:
        ...
class PerfFile:
    def finalize(self) -> None:
        ...
    def save(self, arg0: str) -> None:
        ...
class SessionTCB:
    trace_mode: bool
class Task:
    custom_context_ptr: typing_extensions.CapsuleType
    type: TaskTypes
    @property
    def inputs(self) -> ...:
        ...
    @inputs.setter
    def inputs(self, arg0: ..., std: ...) -> None:
        ...
    @property
    def outputs(self) -> ...:
        ...
    @outputs.setter
    def outputs(self, arg0: ..., std: ...) -> None:
        ...
class TaskTypes:
    """
    Members:
    
      ExecuteOp
    
      ExecuteModule
    """
    ExecuteModule: typing.ClassVar[TaskTypes]  # value = <TaskTypes.ExecuteModule: 1>
    ExecuteOp: typing.ClassVar[TaskTypes]  # value = <TaskTypes.ExecuteOp: 0>
    __members__: typing.ClassVar[dict[str, TaskTypes]]  # value = {'ExecuteOp': <TaskTypes.ExecuteOp: 0>, 'ExecuteModule': <TaskTypes.ExecuteModule: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Tensor:
    @staticmethod
    def arange(start: typing.SupportsFloat, end: typing.SupportsFloat, step: typing.SupportsFloat, dtype: DataTypes = ..., device: DeviceTypes = ...) -> Tensor:
        ...
    @staticmethod
    def empty(shape: ..., std: ..., dtype: DataTypes = ..., device: DeviceTypes = ...) -> Tensor:
        ...
    @staticmethod
    def nil() -> Tensor:
        ...
    @staticmethod
    def ones(shape: ..., std: ..., dtype: DataTypes = ..., device: DeviceTypes = ...) -> Tensor:
        ...
    @staticmethod
    def random(shape: ..., std: ..., start: typing.SupportsFloat = -1.0, end: typing.SupportsFloat = 1.0, dtype: DataTypes = ..., device: DeviceTypes = ...) -> Tensor:
        ...
    @staticmethod
    def zeros(shape: ..., std: ..., dtype: DataTypes = ..., device: DeviceTypes = ...) -> Tensor:
        ...
    def T(self) -> Tensor:
        ...
    @typing.overload
    def __add__(self, arg0: Tensor) -> Tensor:
        ...
    @typing.overload
    def __add__(self, arg0: typing.SupportsFloat) -> Tensor:
        ...
    def __bool__(self) -> bool:
        ...
    def __init__(self) -> None:
        ...
    @typing.overload
    def __mul__(self, arg0: Tensor) -> Tensor:
        ...
    @typing.overload
    def __mul__(self, arg0: typing.SupportsFloat) -> Tensor:
        ...
    def __neg__(self) -> Tensor:
        ...
    @typing.overload
    def __sub__(self, arg0: Tensor) -> Tensor:
        ...
    @typing.overload
    def __sub__(self, arg0: typing.SupportsFloat) -> Tensor:
        ...
    @typing.overload
    def __truediv__(self, arg0: Tensor) -> Tensor:
        ...
    @typing.overload
    def __truediv__(self, arg0: typing.SupportsFloat) -> Tensor:
        ...
    def alloc(self) -> Tensor:
        ...
    def alloc_extra_tensor_view(self, extra_tensor_name: str, shape: ..., std: ..., dtype: DataTypes = ..., device: DeviceTypes = ...) -> Tensor:
        ...
    def bytes(self) -> int:
        ...
    def clone(self) -> Tensor:
        ...
    def contiguous(self) -> Tensor:
        ...
    def cpu(self) -> Tensor:
        ...
    def cuda(self) -> Tensor:
        ...
    def device(self) -> DeviceTypes:
        ...
    def dtype(self) -> DataTypes:
        ...
    def get_extra_tensor_view_in_tensor(self, arg0: str) -> Tensor:
        ...
    def is_contiguous(self) -> bool:
        ...
    def is_nil(self) -> bool:
        ...
    def max(self, keep_dim: bool = False, dim: typing.SupportsInt = 2147483647) -> Tensor:
        ...
    def mem_type(self) -> ...:
        ...
    def min(self, keep_dim: bool = False, dim: typing.SupportsInt = 2147483647) -> Tensor:
        ...
    def name(self) -> str:
        ...
    def numel(self) -> int:
        ...
    def permute(self, arg0: ..., std: ...) -> Tensor:
        ...
    def repeat(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> Tensor:
        ...
    def reshape(self, arg0: ..., std: ...) -> Tensor:
        ...
    def set_mem_type(self, arg0: ...) -> Tensor:
        ...
    def set_name(self, arg0: str) -> Tensor:
        ...
    def shape(self) -> ...:
        ...
    def stride(self) -> ...:
        ...
    def sum(self, keep_dim: bool = False, dim: typing.SupportsInt = 2147483647) -> Tensor:
        ...
    @typing.overload
    def to(self, arg0: DeviceTypes) -> Tensor:
        ...
    @typing.overload
    def to(self, arg0: DataTypes) -> Tensor:
        ...
    def transpose(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> Tensor:
        ...
    def unsqueeze(self, arg0: typing.SupportsInt) -> Tensor:
        ...
    def uuid(self) -> int:
        ...
    def view(self, arg0: ..., std: ...) -> Tensor:
        ...
def clean_this_thread() -> None:
    """
    Clean current thread context
    """
def get_perf_file() -> PerfFile:
    """
    Get performance file
    """
def initialize_context() -> None:
    """
    Initialize the MLLM context
    """
def is_opencl_available() -> bool:
    """
    Check if OpenCL is available
    """
def is_qnn_available() -> bool:
    """
    Check if QNN is available
    """
def load(file_name: str, version: ModelFileVersion = ..., map_2_device: DeviceTypes = ...) -> ...:
    """
    Load parameter file
    """
def memory_report() -> None:
    """
    Print memory report
    """
def perf_end() -> None:
    """
    End performance profiling
    """
def perf_start() -> None:
    """
    Start performance profiling
    """
def save(file_name: str, parameter_file: ..., version: ModelFileVersion = ..., map_2_device: DeviceTypes = ...) -> None:
    """
    Save parameter file
    """
def set_maximum_num_threads(num_threads: typing.SupportsInt) -> None:
    """
    Set maximum number of threads
    """
def set_random_seed(seed: typing.SupportsInt) -> None:
    """
    Set random seed
    """
def shutdown_context() -> None:
    """
    Shutdown the MLLM context
    """
def this_thread() -> SessionTCB:
    """
    Get current thread context
    """
