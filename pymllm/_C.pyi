from __future__ import annotations
import collections.abc
import typing
import typing_extensions
__all__: list[str] = ['AbstractNnNode', 'Backend', 'BaseOp', 'BaseOpOptionsBase', 'CXXLayer', 'CXXModule', 'CausalMaskOpOptions', 'ConfigFile', 'Context', 'DataTypes', 'DeviceTypes', 'Dispatcher', 'DispatcherManager', 'DispatcherManagerOptions', 'EmbeddingOpOptions', 'GELUOpOptions', 'KVCacheOpOptions', 'LayerImpl', 'LayerNormOpOptions', 'LinearImplTypes', 'LinearOp', 'LinearOpOptions', 'MemoryManager', 'MemoryManagerOptions', 'ModelFileVersion', 'ModuleImpl', 'OpTypes', 'ParameterFile', 'RMSNormOpOptions', 'SessionTCB', 'SiLUOpOptions', 'SoftmaxOpOptions', 'Task', 'TaskTypes', 'Tensor', 'TensorMemTypes', 'clean_this_thread', 'initialize_context', 'is_opencl_available', 'is_qnn_available', 'load', 'memory_report', 'save', 'set_maximum_num_threads', 'set_random_seed', 'shutdown_context', 'this_thread']
class AbstractNnNode:
    def depth_decrease(self) -> None:
        ...
    def depth_increase(self) -> None:
        ...
    def get_absolute_name(self) -> str:
        ...
    def get_depth(self) -> int:
        ...
    def get_device(self) -> DeviceTypes:
        ...
    def get_name(self) -> str:
        ...
    def get_type(self) -> ...:
        ...
    def is_compiled_as_obj(self) -> bool:
        ...
    def ref_child_nodes(self) -> list[AbstractNnNode]:
        ...
    def ref_parent_node(self) -> ...:
        ...
    def reg_child_node(self, arg0: AbstractNnNode) -> None:
        ...
    def set_absolute_name(self, arg0: str) -> None:
        ...
    def set_compiled_as_obj(self, arg0: bool) -> None:
        ...
    def set_depth(self, arg0: typing.SupportsInt) -> None:
        ...
    def set_name(self, arg0: str) -> None:
        ...
class Backend:
    def create_op(self, arg0: OpTypes, arg1: BaseOpOptionsBase) -> BaseOp:
        ...
class BaseOp:
    def forward(self, arg0: collections.abc.Sequence[Tensor], arg1: collections.abc.Sequence[Tensor]) -> None:
        ...
    def get_device(self) -> DeviceTypes:
        ...
    def get_name(self) -> str:
        ...
    def get_op_type(self) -> OpTypes:
        ...
    def get_params(self) -> ParameterFile:
        ...
    def load(self, arg0: ParameterFile) -> None:
        ...
    def reshape(self, arg0: collections.abc.Sequence[Tensor], arg1: collections.abc.Sequence[Tensor]) -> None:
        ...
    def set_device_type(self, arg0: DeviceTypes) -> None:
        ...
    def set_name(self, arg0: str) -> None:
        ...
    def setup(self, arg0: collections.abc.Sequence[Tensor], arg1: collections.abc.Sequence[Tensor]) -> None:
        ...
    def trace(self, arg0: typing_extensions.CapsuleType, arg1: collections.abc.Sequence[Tensor], arg2: collections.abc.Sequence[Tensor]) -> None:
        ...
class BaseOpOptionsBase:
    pass
class CXXLayer:
    def __init__(self, impl: LayerImpl) -> None:
        ...
    def forward(self, arg0: collections.abc.Sequence[Tensor]) -> list[Tensor]:
        ...
class CXXModule:
    def __init__(self, impl: ModuleImpl) -> None:
        ...
    def __send_graph_begin(self, arg0: collections.abc.Sequence[Tensor]) -> None:
        ...
    def __send_graph_end(self, arg0: collections.abc.Sequence[Tensor]) -> None:
        ...
    def __trace(self, arg0: collections.abc.Sequence[Tensor], arg1: collections.abc.Sequence[...]) -> list[Tensor]:
        ...
class CausalMaskOpOptions:
    sliding_window: bool
    def __init__(self) -> None:
        ...
    @property
    def window_size(self) -> int:
        ...
    @window_size.setter
    def window_size(self, arg0: typing.SupportsInt) -> None:
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
    def load_string(self, arg0: str) -> None:
        ...
    def save(self, arg0: str) -> None:
        ...
class Context:
    @staticmethod
    def instance() -> Context:
        ...
    def dispatcher_manager(self) -> DispatcherManager:
        ...
    def get_backend(self, arg0: DeviceTypes) -> Backend:
        ...
    def get_random_seed(self) -> int:
        ...
    def get_uuid(self) -> int:
        ...
    def main_thread(self) -> SessionTCB:
        ...
    def memory_manager(self) -> MemoryManager:
        ...
    def ref_session_threads(self) -> dict[..., SessionTCB]:
        ...
    def set_random_seed(self, arg0: typing.SupportsInt) -> None:
        ...
    def this_thread(self) -> SessionTCB:
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
    def sync_wait(self, arg0: typing.SupportsInt) -> None:
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
class EmbeddingOpOptions:
    def __init__(self) -> None:
        ...
    @property
    def hidden_size(self) -> int:
        ...
    @hidden_size.setter
    def hidden_size(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def vocab_size(self) -> int:
        ...
    @vocab_size.setter
    def vocab_size(self, arg0: typing.SupportsInt) -> None:
        ...
class GELUOpOptions:
    def __init__(self) -> None:
        ...
class KVCacheOpOptions:
    use_fa2: bool
    def __init__(self) -> None:
        ...
    @property
    def head_dim(self) -> int:
        ...
    @head_dim.setter
    def head_dim(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def kv_head(self) -> int:
        ...
    @kv_head.setter
    def kv_head(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def layer_idx(self) -> int:
        ...
    @layer_idx.setter
    def layer_idx(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def q_head(self) -> int:
        ...
    @q_head.setter
    def q_head(self, arg0: typing.SupportsInt) -> None:
        ...
class LayerImpl(AbstractNnNode):
    def __init__(self, op_type: OpTypes, options: LinearOpOptions) -> None:
        ...
    def get_instanced_op(self) -> BaseOp:
        ...
    def load(self, arg0: ParameterFile) -> None:
        ...
    def op_type(self) -> OpTypes:
        ...
    def ref_options(self) -> BaseOpOptionsBase:
        ...
    def set_instanced_op(self, arg0: BaseOp) -> None:
        ...
    def to(self, arg0: DeviceTypes) -> None:
        ...
class LayerNormOpOptions:
    bias: bool
    elementwise_affine: bool
    def __init__(self) -> None:
        ...
    @property
    def eps(self) -> float:
        ...
    @eps.setter
    def eps(self, arg0: typing.SupportsFloat) -> None:
        ...
    @property
    def normalized_shape(self) -> list[int]:
        ...
    @normalized_shape.setter
    def normalized_shape(self, arg0: collections.abc.Sequence[typing.SupportsInt]) -> None:
        ...
class LinearImplTypes:
    """
    Members:
    
      LinearImplTypes_Start
    
      Default
    
      Kleidiai_Start
    
      Kleidiai_End
    
      GGUF_Start
    
      GGUF_End
    
      LinearImplTypes_End
    """
    Default: typing.ClassVar[LinearImplTypes]  # value = <LinearImplTypes.Default: 1>
    GGUF_End: typing.ClassVar[LinearImplTypes]  # value = <LinearImplTypes.GGUF_End: 16>
    GGUF_Start: typing.ClassVar[LinearImplTypes]  # value = <LinearImplTypes.GGUF_Start: 15>
    Kleidiai_End: typing.ClassVar[LinearImplTypes]  # value = <LinearImplTypes.Kleidiai_End: 14>
    Kleidiai_Start: typing.ClassVar[LinearImplTypes]  # value = <LinearImplTypes.Kleidiai_Start: 3>
    LinearImplTypes_End: typing.ClassVar[LinearImplTypes]  # value = <LinearImplTypes.LinearImplTypes_End: 17>
    LinearImplTypes_Start: typing.ClassVar[LinearImplTypes]  # value = <LinearImplTypes.LinearImplTypes_Start: 0>
    __members__: typing.ClassVar[dict[str, LinearImplTypes]]  # value = {'LinearImplTypes_Start': <LinearImplTypes.LinearImplTypes_Start: 0>, 'Default': <LinearImplTypes.Default: 1>, 'Kleidiai_Start': <LinearImplTypes.Kleidiai_Start: 3>, 'Kleidiai_End': <LinearImplTypes.Kleidiai_End: 14>, 'GGUF_Start': <LinearImplTypes.GGUF_Start: 15>, 'GGUF_End': <LinearImplTypes.GGUF_End: 16>, 'LinearImplTypes_End': <LinearImplTypes.LinearImplTypes_End: 17>}
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
class LinearOp(BaseOp):
    def __init__(self, arg0: LinearOpOptions) -> None:
        ...
    def bias(self) -> Tensor:
        ...
    def forward(self, arg0: collections.abc.Sequence[Tensor], arg1: collections.abc.Sequence[Tensor]) -> None:
        ...
    def load(self, arg0: ParameterFile) -> None:
        ...
    def options(self) -> LinearOpOptions:
        ...
    def reshape(self, arg0: collections.abc.Sequence[Tensor], arg1: collections.abc.Sequence[Tensor]) -> None:
        ...
    def setup(self, arg0: collections.abc.Sequence[Tensor], arg1: collections.abc.Sequence[Tensor]) -> None:
        ...
    def weight(self) -> Tensor:
        ...
class LinearOpOptions:
    bias: bool
    impl_type: LinearImplTypes
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, in_channels: typing.SupportsInt = 0, out_channels: typing.SupportsInt = 0, bias: bool = True, impl_type: LinearImplTypes = ...) -> None:
        ...
    def set_inputs_dtype(self, arg0: typing.SupportsInt, arg1: DataTypes) -> LinearOpOptions:
        ...
    def set_outputs_dtype(self, arg0: typing.SupportsInt, arg1: DataTypes) -> LinearOpOptions:
        ...
    @property
    def in_channels(self) -> int:
        ...
    @in_channels.setter
    def in_channels(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def out_channels(self) -> int:
        ...
    @out_channels.setter
    def out_channels(self, arg0: typing.SupportsInt) -> None:
        ...
class MemoryManager:
    def clear_all(self) -> None:
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
    def load(self, arg0: ParameterFile) -> None:
        ...
    def params(self, arg0: ModelFileVersion) -> ParameterFile:
        ...
    def to(self, arg0: DeviceTypes) -> None:
        ...
class OpTypes:
    """
    Members:
    
      OpType_Start
    
      Fill
    
      Add
    
      Sub
    
      Mul
    
      Div
    
      MatMul
    
      Embedding
    
      Linear
    
      RoPE
    
      Softmax
    
      Transpose
    
      RMSNorm
    
      SiLU
    
      KVCache
    
      CausalMask
    
      CastType
    
      X2X
    
      Split
    
      View
    
      FlashAttention2
    
      Repeat
    
      Permute
    
      Conv3D
    
      Conv2D
    
      Conv1D
    
      GELU
    
      LayerNorm
    
      MultimodalRoPE
    
      VisionRoPE
    
      QuickGELU
    
      Copy
    
      Clone
    
      Neg
    
      Concat
    
      ReLU
    
      ReLU2
    
      ReduceMax
    
      ReduceMin
    
      ReduceSum
    
      Contiguous
    
      Reshape
    
      GraphBegin
    
      GraphEnd
    
      OpType_End
    """
    Add: typing.ClassVar[OpTypes]  # value = <OpTypes.Add: 2>
    CastType: typing.ClassVar[OpTypes]  # value = <OpTypes.CastType: 18>
    CausalMask: typing.ClassVar[OpTypes]  # value = <OpTypes.CausalMask: 17>
    Clone: typing.ClassVar[OpTypes]  # value = <OpTypes.Clone: 34>
    Concat: typing.ClassVar[OpTypes]  # value = <OpTypes.Concat: 36>
    Contiguous: typing.ClassVar[OpTypes]  # value = <OpTypes.Contiguous: 42>
    Conv1D: typing.ClassVar[OpTypes]  # value = <OpTypes.Conv1D: 27>
    Conv2D: typing.ClassVar[OpTypes]  # value = <OpTypes.Conv2D: 26>
    Conv3D: typing.ClassVar[OpTypes]  # value = <OpTypes.Conv3D: 25>
    Copy: typing.ClassVar[OpTypes]  # value = <OpTypes.Copy: 33>
    Div: typing.ClassVar[OpTypes]  # value = <OpTypes.Div: 5>
    Embedding: typing.ClassVar[OpTypes]  # value = <OpTypes.Embedding: 7>
    Fill: typing.ClassVar[OpTypes]  # value = <OpTypes.Fill: 1>
    FlashAttention2: typing.ClassVar[OpTypes]  # value = <OpTypes.FlashAttention2: 22>
    GELU: typing.ClassVar[OpTypes]  # value = <OpTypes.GELU: 28>
    GraphBegin: typing.ClassVar[OpTypes]  # value = <OpTypes.GraphBegin: 55>
    GraphEnd: typing.ClassVar[OpTypes]  # value = <OpTypes.GraphEnd: 56>
    KVCache: typing.ClassVar[OpTypes]  # value = <OpTypes.KVCache: 16>
    LayerNorm: typing.ClassVar[OpTypes]  # value = <OpTypes.LayerNorm: 29>
    Linear: typing.ClassVar[OpTypes]  # value = <OpTypes.Linear: 8>
    MatMul: typing.ClassVar[OpTypes]  # value = <OpTypes.MatMul: 6>
    Mul: typing.ClassVar[OpTypes]  # value = <OpTypes.Mul: 4>
    MultimodalRoPE: typing.ClassVar[OpTypes]  # value = <OpTypes.MultimodalRoPE: 30>
    Neg: typing.ClassVar[OpTypes]  # value = <OpTypes.Neg: 35>
    OpType_End: typing.ClassVar[OpTypes]  # value = <OpTypes.OpType_End: 57>
    OpType_Start: typing.ClassVar[OpTypes]  # value = <OpTypes.OpType_Start: 0>
    Permute: typing.ClassVar[OpTypes]  # value = <OpTypes.Permute: 24>
    QuickGELU: typing.ClassVar[OpTypes]  # value = <OpTypes.QuickGELU: 32>
    RMSNorm: typing.ClassVar[OpTypes]  # value = <OpTypes.RMSNorm: 14>
    ReLU: typing.ClassVar[OpTypes]  # value = <OpTypes.ReLU: 37>
    ReLU2: typing.ClassVar[OpTypes]  # value = <OpTypes.ReLU2: 38>
    ReduceMax: typing.ClassVar[OpTypes]  # value = <OpTypes.ReduceMax: 39>
    ReduceMin: typing.ClassVar[OpTypes]  # value = <OpTypes.ReduceMin: 40>
    ReduceSum: typing.ClassVar[OpTypes]  # value = <OpTypes.ReduceSum: 41>
    Repeat: typing.ClassVar[OpTypes]  # value = <OpTypes.Repeat: 23>
    Reshape: typing.ClassVar[OpTypes]  # value = <OpTypes.Reshape: 43>
    RoPE: typing.ClassVar[OpTypes]  # value = <OpTypes.RoPE: 9>
    SiLU: typing.ClassVar[OpTypes]  # value = <OpTypes.SiLU: 15>
    Softmax: typing.ClassVar[OpTypes]  # value = <OpTypes.Softmax: 10>
    Split: typing.ClassVar[OpTypes]  # value = <OpTypes.Split: 20>
    Sub: typing.ClassVar[OpTypes]  # value = <OpTypes.Sub: 3>
    Transpose: typing.ClassVar[OpTypes]  # value = <OpTypes.Transpose: 13>
    View: typing.ClassVar[OpTypes]  # value = <OpTypes.View: 21>
    VisionRoPE: typing.ClassVar[OpTypes]  # value = <OpTypes.VisionRoPE: 31>
    X2X: typing.ClassVar[OpTypes]  # value = <OpTypes.X2X: 19>
    __members__: typing.ClassVar[dict[str, OpTypes]]  # value = {'OpType_Start': <OpTypes.OpType_Start: 0>, 'Fill': <OpTypes.Fill: 1>, 'Add': <OpTypes.Add: 2>, 'Sub': <OpTypes.Sub: 3>, 'Mul': <OpTypes.Mul: 4>, 'Div': <OpTypes.Div: 5>, 'MatMul': <OpTypes.MatMul: 6>, 'Embedding': <OpTypes.Embedding: 7>, 'Linear': <OpTypes.Linear: 8>, 'RoPE': <OpTypes.RoPE: 9>, 'Softmax': <OpTypes.Softmax: 10>, 'Transpose': <OpTypes.Transpose: 13>, 'RMSNorm': <OpTypes.RMSNorm: 14>, 'SiLU': <OpTypes.SiLU: 15>, 'KVCache': <OpTypes.KVCache: 16>, 'CausalMask': <OpTypes.CausalMask: 17>, 'CastType': <OpTypes.CastType: 18>, 'X2X': <OpTypes.X2X: 19>, 'Split': <OpTypes.Split: 20>, 'View': <OpTypes.View: 21>, 'FlashAttention2': <OpTypes.FlashAttention2: 22>, 'Repeat': <OpTypes.Repeat: 23>, 'Permute': <OpTypes.Permute: 24>, 'Conv3D': <OpTypes.Conv3D: 25>, 'Conv2D': <OpTypes.Conv2D: 26>, 'Conv1D': <OpTypes.Conv1D: 27>, 'GELU': <OpTypes.GELU: 28>, 'LayerNorm': <OpTypes.LayerNorm: 29>, 'MultimodalRoPE': <OpTypes.MultimodalRoPE: 30>, 'VisionRoPE': <OpTypes.VisionRoPE: 31>, 'QuickGELU': <OpTypes.QuickGELU: 32>, 'Copy': <OpTypes.Copy: 33>, 'Clone': <OpTypes.Clone: 34>, 'Neg': <OpTypes.Neg: 35>, 'Concat': <OpTypes.Concat: 36>, 'ReLU': <OpTypes.ReLU: 37>, 'ReLU2': <OpTypes.ReLU2: 38>, 'ReduceMax': <OpTypes.ReduceMax: 39>, 'ReduceMin': <OpTypes.ReduceMin: 40>, 'ReduceSum': <OpTypes.ReduceSum: 41>, 'Contiguous': <OpTypes.Contiguous: 42>, 'Reshape': <OpTypes.Reshape: 43>, 'GraphBegin': <OpTypes.GraphBegin: 55>, 'GraphEnd': <OpTypes.GraphEnd: 56>, 'OpType_End': <OpTypes.OpType_End: 57>}
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
class ParameterFile:
    def __init__(self, v: ModelFileVersion = ...) -> None:
        ...
    def has(self, arg0: str) -> bool:
        ...
    def pull(self, arg0: str) -> Tensor:
        ...
    def push(self, arg0: str, arg1: Tensor) -> None:
        ...
    def remove(self, arg0: str) -> None:
        ...
class RMSNormOpOptions:
    add_unit_offset: bool
    def __init__(self) -> None:
        ...
    @property
    def epsilon(self) -> float:
        ...
    @epsilon.setter
    def epsilon(self, arg0: typing.SupportsFloat) -> None:
        ...
class SessionTCB:
    trace_mode: bool
class SiLUOpOptions:
    def __init__(self) -> None:
        ...
class SoftmaxOpOptions:
    def __init__(self) -> None:
        ...
    @property
    def axis(self) -> int:
        ...
    @axis.setter
    def axis(self, arg0: typing.SupportsInt) -> None:
        ...
class Task:
    custom_context_ptr: typing_extensions.CapsuleType
    type: TaskTypes
    @property
    def inputs(self) -> list[Tensor]:
        ...
    @inputs.setter
    def inputs(self, arg0: collections.abc.Sequence[Tensor]) -> None:
        ...
    @property
    def outputs(self) -> list[Tensor]:
        ...
    @outputs.setter
    def outputs(self, arg0: collections.abc.Sequence[Tensor]) -> None:
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
    def empty(shape: collections.abc.Sequence[typing.SupportsInt], dtype: DataTypes = ..., device: DeviceTypes = ...) -> Tensor:
        ...
    @staticmethod
    def nil() -> Tensor:
        ...
    @staticmethod
    def ones(shape: collections.abc.Sequence[typing.SupportsInt], dtype: DataTypes = ..., device: DeviceTypes = ...) -> Tensor:
        ...
    @staticmethod
    def random(shape: collections.abc.Sequence[typing.SupportsInt], start: typing.SupportsFloat = -1.0, end: typing.SupportsFloat = 1.0, dtype: DataTypes = ..., device: DeviceTypes = ...) -> Tensor:
        ...
    @staticmethod
    def zeros(shape: collections.abc.Sequence[typing.SupportsInt], dtype: DataTypes = ..., device: DeviceTypes = ...) -> Tensor:
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
    def __str__(self) -> str:
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
    def alloc_extra_tensor_view(self, extra_tensor_name: str, shape: collections.abc.Sequence[typing.SupportsInt], dtype: DataTypes = ..., device: DeviceTypes = ...) -> Tensor:
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
    def max(self, keep_dim: typing.SupportsInt = False, dim: bool = 2147483647) -> Tensor:
        ...
    def mem_type(self) -> TensorMemTypes:
        ...
    def min(self, keep_dim: typing.SupportsInt = False, dim: bool = 2147483647) -> Tensor:
        ...
    def name(self) -> str:
        ...
    def numel(self) -> int:
        ...
    def permute(self, arg0: collections.abc.Sequence[typing.SupportsInt]) -> Tensor:
        ...
    def repeat(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> Tensor:
        ...
    def reshape(self, arg0: collections.abc.Sequence[typing.SupportsInt]) -> Tensor:
        ...
    def set_mem_type(self, arg0: TensorMemTypes) -> Tensor:
        ...
    def set_name(self, arg0: str) -> Tensor:
        ...
    def shape(self) -> list[int]:
        ...
    def stride(self) -> list[int]:
        ...
    def sum(self, keep_dim: typing.SupportsInt = False, dim: bool = 2147483647) -> Tensor:
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
    def view(self, arg0: collections.abc.Sequence[typing.SupportsInt]) -> Tensor:
        ...
class TensorMemTypes:
    """
    Members:
    
      TensorMemTypes_Start
    
      Normal
    
      ExtraInput
    
      ExtraOutput
    
      Manual
    
      Global
    
      Params_Start
    
      ParamsMMAP
    
      ParamsNormal
    
      Params_End
    
      QnnAppRead
    
      QnnAppWrite
    
      QnnAppReadWrite
    
      TensorMemTypes_End
    """
    ExtraInput: typing.ClassVar[TensorMemTypes]  # value = <TensorMemTypes.ExtraInput: 2>
    ExtraOutput: typing.ClassVar[TensorMemTypes]  # value = <TensorMemTypes.ExtraOutput: 3>
    Global: typing.ClassVar[TensorMemTypes]  # value = <TensorMemTypes.Global: 5>
    Manual: typing.ClassVar[TensorMemTypes]  # value = <TensorMemTypes.Manual: 4>
    Normal: typing.ClassVar[TensorMemTypes]  # value = <TensorMemTypes.Normal: 1>
    ParamsMMAP: typing.ClassVar[TensorMemTypes]  # value = <TensorMemTypes.ParamsMMAP: 7>
    ParamsNormal: typing.ClassVar[TensorMemTypes]  # value = <TensorMemTypes.ParamsNormal: 8>
    Params_End: typing.ClassVar[TensorMemTypes]  # value = <TensorMemTypes.Params_End: 9>
    Params_Start: typing.ClassVar[TensorMemTypes]  # value = <TensorMemTypes.Params_Start: 6>
    QnnAppRead: typing.ClassVar[TensorMemTypes]  # value = <TensorMemTypes.QnnAppRead: 10>
    QnnAppReadWrite: typing.ClassVar[TensorMemTypes]  # value = <TensorMemTypes.QnnAppReadWrite: 12>
    QnnAppWrite: typing.ClassVar[TensorMemTypes]  # value = <TensorMemTypes.QnnAppWrite: 11>
    TensorMemTypes_End: typing.ClassVar[TensorMemTypes]  # value = <TensorMemTypes.TensorMemTypes_End: 13>
    TensorMemTypes_Start: typing.ClassVar[TensorMemTypes]  # value = <TensorMemTypes.TensorMemTypes_Start: 0>
    __members__: typing.ClassVar[dict[str, TensorMemTypes]]  # value = {'TensorMemTypes_Start': <TensorMemTypes.TensorMemTypes_Start: 0>, 'Normal': <TensorMemTypes.Normal: 1>, 'ExtraInput': <TensorMemTypes.ExtraInput: 2>, 'ExtraOutput': <TensorMemTypes.ExtraOutput: 3>, 'Manual': <TensorMemTypes.Manual: 4>, 'Global': <TensorMemTypes.Global: 5>, 'Params_Start': <TensorMemTypes.Params_Start: 6>, 'ParamsMMAP': <TensorMemTypes.ParamsMMAP: 7>, 'ParamsNormal': <TensorMemTypes.ParamsNormal: 8>, 'Params_End': <TensorMemTypes.Params_End: 9>, 'QnnAppRead': <TensorMemTypes.QnnAppRead: 10>, 'QnnAppWrite': <TensorMemTypes.QnnAppWrite: 11>, 'QnnAppReadWrite': <TensorMemTypes.QnnAppReadWrite: 12>, 'TensorMemTypes_End': <TensorMemTypes.TensorMemTypes_End: 13>}
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
def clean_this_thread() -> None:
    """
    Clean current thread context
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
def load(file_name: str, version: ModelFileVersion = ..., map_2_device: DeviceTypes = ...) -> ParameterFile:
    """
    Load parameter file
    """
def memory_report() -> None:
    """
    Print memory report
    """
def save(file_name: str, parameter_file: ParameterFile, version: ModelFileVersion = ..., map_2_device: DeviceTypes = ...) -> None:
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
