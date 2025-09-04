"""
Functional interface for MLLM operations.
"""

from typing import List, Union, Tuple
from .._C import Context, OpTypes
from .._C import Tensor
from .._C import BaseOpOptionsBase
from .._C import (
    MatMulOpOptions,
    ViewOpOptions,
    SplitOpOptions,
    ConcatOpOptions,
    SoftmaxOpOptions,
    LogOpOptions,
    ExpOpOptions,
    SinOpOptions,
    CosOpOptions,
    TopKOpOptions,
    ClipOpOptions,
    ReduceMinOpOptions,
    ReduceMaxOpOptions,
    ReduceSumOpOptions,
    MeanOpOptions,
)


def matmul(
    A: Tensor, B: Tensor, transpose_A: bool = False, transpose_B: bool = False
) -> Tensor:
    """
    Matrix multiplication of two tensors.

    Args:
        A: First tensor
        B: Second tensor
        transpose_A: Whether to transpose A before multiplication
        transpose_B: Whether to transpose B before multiplication

    Returns:
        Result tensor of matrix multiplication
    """
    options = MatMulOpOptions()
    options.transpose_a = transpose_A
    options.transpose_b = transpose_B
    options.matmul_type = 0  # kDefault

    outputs = Context.instance().build_op_and_submit_task(
        OpTypes.MatMul, options, [A, B]
    )
    return outputs[0]


def view(x: Tensor, shape: List[int]) -> Tensor:
    """
    Reshape tensor to specified shape.

    Args:
        x: Input tensor
        shape: Target shape

    Returns:
        Reshaped tensor
    """
    options = ViewOpOptions()
    options.to_shape = shape

    outputs = Context.instance().build_op_and_submit_task(OpTypes.View, options, [x])
    return outputs[0]


def split(
    x: Tensor, split_size_or_sections: Union[int, List[int]], dim: int = 0
) -> List[Tensor]:
    """
    Split tensor into chunks.

    Args:
        x: Input tensor
        split_size_or_sections: Size of each chunk or list of sizes
        dim: Dimension along which to split

    Returns:
        List of split tensors
    """
    options = SplitOpOptions()
    options.dim = dim
    if isinstance(split_size_or_sections, int):
        options.split_size_or_sections = [split_size_or_sections]
    else:
        options.split_size_or_sections = split_size_or_sections

    outputs = Context.instance().build_op_and_submit_task(OpTypes.Split, options, [x])
    return outputs


def concat(tensors: List[Tensor], dim: int) -> Tensor:
    """
    Concatenate tensors along specified dimension.

    Args:
        tensors: List of tensors to concatenate
        dim: Dimension along which to concatenate

    Returns:
        Concatenated tensor
    """
    options = ConcatOpOptions()
    options.dim = dim

    outputs = Context.instance().build_op_and_submit_task(
        OpTypes.Concat, options, tensors
    )
    return outputs[0]


def softmax(x: Tensor, dim: int) -> Tensor:
    """
    Apply softmax along specified dimension.

    Args:
        x: Input tensor
        dim: Dimension along which to apply softmax

    Returns:
        Tensor with softmax applied
    """
    options = SoftmaxOpOptions()
    options.axis = dim

    outputs = Context.instance().build_op_and_submit_task(OpTypes.Softmax, options, [x])
    return outputs[0]


def log(x: Tensor) -> Tensor:
    """
    Element-wise natural logarithm.

    Args:
        x: Input tensor

    Returns:
        Tensor with natural logarithm applied
    """
    options = LogOpOptions()

    outputs = Context.instance().build_op_and_submit_task(OpTypes.Log, options, [x])
    return outputs[0]


def exp(x: Tensor) -> Tensor:
    """
    Element-wise exponential.

    Args:
        x: Input tensor

    Returns:
        Tensor with exponential applied
    """
    options = ExpOpOptions()

    outputs = Context.instance().build_op_and_submit_task(OpTypes.Exp, options, [x])
    return outputs[0]


def sin(x: Tensor) -> Tensor:
    """
    Element-wise sine.

    Args:
        x: Input tensor

    Returns:
        Tensor with sine applied
    """
    options = SinOpOptions()

    outputs = Context.instance().build_op_and_submit_task(OpTypes.Sin, options, [x])
    return outputs[0]


def cos(x: Tensor) -> Tensor:
    """
    Element-wise cosine.

    Args:
        x: Input tensor

    Returns:
        Tensor with cosine applied
    """
    options = CosOpOptions()

    outputs = Context.instance().build_op_and_submit_task(OpTypes.Cos, options, [x])
    return outputs[0]


def topk(
    x: Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True
) -> Tuple[Tensor, Tensor]:
    """
    Returns the k largest elements of the given input tensor along a given dimension.

    Args:
        x: Input tensor
        k: The k in "top-k"
        dim: The dimension to sort along
        largest: Controls whether to return largest or smallest elements
        sorted: Controls whether to return the elements in sorted order

    Returns:
        Tuple of (values, indices)
    """
    options = TopKOpOptions()
    options.k = k
    options.dim = dim
    options.largest = largest
    options.sorted = sorted

    outputs = Context.instance().build_op_and_submit_task(OpTypes.TopK, options, [x])
    return (outputs[0], outputs[1])


def clip(x: Tensor, min_val: float, max_val: float) -> Tensor:
    """
    Clip tensor values to be within [min_val, max_val].

    Args:
        x: Input tensor
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clipped tensor
    """
    options = ClipOpOptions()
    options.min_val = min_val
    options.max_val = max_val

    outputs = Context.instance().build_op_and_submit_task(
        OpTypes.Clip,
        BaseOpOptionsBase(options),
        [x],
        x.device(),
    )
    return outputs[0]


def min(x: Tensor, dim: int = None, keep_dim: bool = False) -> Tensor:
    """
    Returns the minimum value of all elements in the input tensor.

    Args:
        x: Input tensor
        dim: The dimension to reduce. If None, reduce all dimensions
        keep_dim: Whether the output tensor has dim retained or not

    Returns:
        Minimum values tensor
    """
    options = ReduceMinOpOptions()
    if dim is None:
        options.dim = 2147483647  # std::numeric_limits<int32_t>::max()
    else:
        options.dim = dim
    options.keep_dim = keep_dim

    outputs = Context.instance().build_op_and_submit_task(
        OpTypes.ReduceMin, options, [x]
    )
    return outputs[0]


def max(x: Tensor, dim: int = None, keep_dim: bool = False) -> Tensor:
    """
    Returns the maximum value of all elements in the input tensor.

    Args:
        x: Input tensor
        dim: The dimension to reduce. If None, reduce all dimensions
        keep_dim: Whether the output tensor has dim retained or not

    Returns:
        Maximum values tensor
    """
    options = ReduceMaxOpOptions()
    if dim is None:
        options.dim = 2147483647  # std::numeric_limits<int32_t>::max()
    else:
        options.dim = dim
    options.keep_dim = keep_dim

    outputs = Context.instance().build_op_and_submit_task(
        OpTypes.ReduceMax, options, [x]
    )
    return outputs[0]


def sum(x: Tensor, dim: int = None, keep_dim: bool = False) -> Tensor:
    """
    Returns the sum of all elements in the input tensor.

    Args:
        x: Input tensor
        dim: The dimension to reduce. If None, reduce all dimensions
        keep_dim: Whether the output tensor has dim retained or not

    Returns:
        Sum tensor
    """
    options = ReduceSumOpOptions()
    if dim is None:
        options.dim = 2147483647  # std::numeric_limits<int32_t>::max()
    else:
        options.dim = dim
    options.keep_dim = keep_dim

    outputs = Context.instance().build_op_and_submit_task(
        OpTypes.ReduceSum, options, [x]
    )
    return outputs[0]


def mean(x: Tensor, dim: int = None, keep_dim: bool = False) -> Tensor:
    """
    Returns the mean value of all elements in the input tensor.

    Args:
        x: Input tensor
        dim: The dimension to reduce. If None, reduce all dimensions
        keep_dim: Whether the output tensor has dim retained or not

    Returns:
        Mean tensor
    """
    options = MeanOpOptions()
    if dim is None:
        options.dim = 2147483647  # std::numeric_limits<int32_t>::max()
    else:
        options.dim = dim
    options.keep_dim = keep_dim

    outputs = Context.instance().build_op_and_submit_task(OpTypes.Mean, options, [x])
    return outputs[0]
