MLLM IR
=======

MLLM IR (Intermediate Representation) is a multi-level intermediate representation designed for the MLLM framework. It provides a structured way to represent machine learning models and operations at different abstraction levels, enabling efficient compilation and execution.

Basic IR Types
--------------

MLLM IR consists of several levels, each serving specific purposes in the compilation pipeline:

1. Tensor IR
^^^^^^^^^^^^

Tensor IR represents the lowest level of abstraction, focusing on tensor operations and memory management. It handles fundamental tensor operations and memory allocation/deallocation.

**Operations (Ops):**

- ``RegisterOp``: Registers a tensor in the IR context, typically for parameter tensors or global tensors
- ``AllocOp``: Allocates memory for a tensor during execution
- ``FreeOp``: Frees previously allocated memory for a tensor

**Values:**

- ``TensorValue``: Represents a tensor with its shape, data type, and device information

2. Linalg IR
^^^^^^^^^^^^

Linalg IR represents linear algebra operations commonly found in neural networks. These operations are closer to actual ML computations.

**Operations (Ops):**

- Arithmetic operations: ``AddOp``, ``SubOp``, ``MulOp``, ``DivOp``, ``NegOp``
- Matrix operations: ``MatMulOp``
- Neural network operations: ``EmbeddingOp``, ``LinearOp``, ``RoPEOp``, ``KVCacheOp``, ``CausalMaskOp``, ``SoftmaxOp``
- Normalization operations: ``RMSNormOp``, ``LayerNormOp``
- Activation functions: ``SiLUOp``, ``GELUOp``, ``QuickGELUOp``, ``ReLUOp``
- Data manipulation: ``TransposeOp``, ``PermuteOp``, ``ViewOp``, ``ReshapeOp``, ``SplitOp``, ``ConcatOp``, ``RepeatOp``, ``CastTypeOp``, ``ContiguousOp``
- Memory operations: ``CopyOp``, ``CloneOp``
- Reduction operations: ``ReduceMaxOp``, ``ReduceMinOp``, ``ReduceSumOp``
- Convolution operations: ``Conv1DOp``, ``Conv2DOp``, ``Conv3DOp``
- Attention operations: ``FlashAttention2Op``

**Values:**

- Operations work with ``TensorValue`` instances from the Tensor IR level

3. Graph IR
^^^^^^^^^^^

Graph IR represents higher-level operations and subgraphs, typically corresponding to neural network layers or modules.

**Operations (Ops):**

- ``SubGraphOp``: Represents a subgraph or module within the computation graph
- ``CallGraphOp``: Represents a call to another graph/subgraph

**Values:**

- Graph IR operates on tensor values passed between subgraphs

4. Program IR
^^^^^^^^^^^^^

Program IR represents the executable program structure, including control flow and program fragments.

**Operations (Ops):**

- ``InstructionOp``: Represents one or more executable instructions that can be fused
- ``FragmentOp``: Represents a program fragment (code, data, or text)
- ``JumpOp``: Represents a jump/branch operation
- ``LabelOp``: Represents a label/target for jumps

**Values:**

- Program IR works with tensor values flowing through program instructions

IR Structure Overview
---------------------

Each IR level follows a similar structure with nodes, operations, and values:

- **Nodes**: The basic building blocks of the IR
- **Operations (Ops)**: Computational or structural operations that transform values
- **Values**: Data that flows between operations (typically tensors)

Each operation can have inputs and outputs, forming a computational graph. The IR also supports attributes and metadata for operations and values.

The multi-level design allows MLLM to perform optimizations at different abstraction levels, from high-level graph transformations to low-level kernel optimizations.