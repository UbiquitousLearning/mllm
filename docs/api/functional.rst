Functional API
==============

The functional module provides a collection of stateless functions that perform common operations on tensors. These functions are the functional counterparts to the Layer classes and can be used directly without creating layer objects.

.. code-block:: cpp

   #include "mllm/nn/Functional.hpp"

namespace mllm::nn::functional

Matrix Operations
-----------------

.. cpp:function:: Tensor mllm::nn::functional::matmul(const Tensor& A, const Tensor& B, bool transpose_A = false, bool transpose_B = false, aops::MatMulOpType type = aops::MatMulOpType::kDefault)

   Perform matrix multiplication of two tensors.

   :param A: First input tensor
   :param B: Second input tensor
   :param transpose_A: Whether to transpose tensor A before multiplication (default: false)
   :param transpose_B: Whether to transpose tensor B before multiplication (default: false)
   :param type: Type of matrix multiplication operation (default: kDefault)
   :return: Result of matrix multiplication

Shape Operations
----------------

.. cpp:function:: Tensor mllm::nn::functional::view(const Tensor& x, const std::vector<int32_t>& shape)

   Reshape a tensor to a new shape.

   :param x: Input tensor
   :param shape: New shape for the tensor
   :return: Reshaped tensor

.. cpp:function:: std::vector<Tensor> mllm::nn::functional::split(const Tensor& x, int32_t split_size_or_sections, int32_t dim)

   Split a tensor into chunks along a given dimension.

   :param x: Input tensor
   :param split_size_or_sections: Size of each chunk or list of sizes for each chunk
   :param dim: Dimension along which to split the tensor
   :return: Vector of split tensors

.. cpp:function:: std::vector<Tensor> mllm::nn::functional::split(const Tensor& x, const std::vector<int32_t>& split_size_or_sections, int32_t dim)

   Split a tensor into chunks with specified sizes along a given dimension.

   :param x: Input tensor
   :param split_size_or_sections: List of sizes for each chunk
   :param dim: Dimension along which to split the tensor
   :return: Vector of split tensors

.. cpp:function:: template<int32_t RET_NUM> std::array<Tensor, RET_NUM> mllm::nn::functional::split(const Tensor& x, int32_t split_size_or_sections, int32_t dim)

   Split a tensor into a fixed number of chunks with same size along a given dimension.

   :param x: Input tensor
   :param split_size_or_sections: Size of each chunk
   :param dim: Dimension along which to split the tensor
   :return: Array of split tensors with fixed size

.. cpp:function:: template<int32_t RET_NUM> std::array<Tensor, RET_NUM> mllm::nn::functional::split(const Tensor& x, const std::vector<int32_t>& split_size_or_sections, int32_t dim)

   Split a tensor into a fixed number of chunks with specified sizes along a given dimension.

   :param x: Input tensor
   :param split_size_or_sections: List of sizes for each chunk
   :param dim: Dimension along which to split the tensor
   :return: Array of split tensors with fixed size

.. cpp:function:: Tensor mllm::nn::functional::concat(const std::vector<Tensor>& ins, int32_t dim)

   Concatenate a sequence of tensors along a given dimension.

   :param ins: Vector of input tensors to concatenate
   :param dim: Dimension along which to concatenate
   :return: Concatenated tensor

.. cpp:function:: Tensor mllm::nn::functional::pad(const Tensor& x, const std::vector<int32_t>& pad, aops::PadMode mode = aops::PadMode::kConstant, float value = 0.0f)

   Pad a tensor along the last N dimensions as specified.

   :param x: Input tensor
   :param pad: Padding sizes ordered from the last dimension to the first, e.g. [last_left, last_right, ..., first_left, first_right]
   :param mode: Padding mode (kConstant, kReflect, kReplicate, kCircular). Default: kConstant
   :param value: Constant value used when mode is kConstant. Default: 0.0
   :return: Padded tensor

.. cpp:function:: Tensor mllm::nn::functional::interpolate(const Tensor& x, const std::vector<int32_t>& size, aops::InterpolateOpMode mode = aops::InterpolateOpMode::kNearest, bool align_corners = false, bool antialias = false)

   Resize a tensor to the target spatial size.

   :param x: Input tensor (supports 1D/2D/3D spatial resizing depending on mode)
   :param size: Target spatial size (e.g., [H_out, W_out] for 2D)
   :param mode: Interpolation mode (kNearest, kLinear, kBilinear, kBicubic, kTrilinear). Default: kNearest
   :param align_corners: Align corners for linear/bilinear/trilinear interpolation. Default: false
      :return: Resized tensor

.. cpp:function:: Tensor mllm::nn::functional::interpolate(const Tensor& x, const std::vector<float>& scale_factor, aops::InterpolateOpMode mode = aops::InterpolateOpMode::kNearest, bool align_corners = false)

   Resize a tensor by scale factors per spatial dimension.

   :param x: Input tensor (supports 1D/2D/3D spatial resizing depending on mode)
   :param scale_factor: Scale factors per spatial dimension (e.g., [sh, sw] for 2D)
   :param mode: Interpolation mode (kNearest, kLinear, kBilinear, kBicubic, kTrilinear). Default: kNearest
   :param align_corners: Align corners for linear/bilinear/trilinear interpolation. Default: false
   :return: Resized tensor

Attention Operations
--------------------

.. cpp:function:: Tensor mllm::nn::functional::flashAttention2(const Tensor& Q, const Tensor& K, const Tensor& V)

   Perform FlashAttention-2 operation on query, key, and value tensors.

   :param Q: Query tensor in BSHD format
   :param K: Key tensor in BSHD format
   :param V: Value tensor in BSHD format
   :return: Output tensor after attention operation

Activation Functions
--------------------

.. cpp:function:: Tensor mllm::nn::functional::softmax(const Tensor& x, int32_t dim)

   Apply softmax activation function along a given dimension.

   :param x: Input tensor
   :param dim: Dimension along which to apply softmax
   :return: Tensor with softmax applied

.. cpp:function:: Tensor mllm::nn::functional::log(const Tensor& x)

   Compute natural logarithm of elements in the tensor.

   :param x: Input tensor
   :return: Tensor with natural logarithm applied element-wise

Selection Operations
--------------------

.. cpp:function:: std::array<Tensor, 2> mllm::nn::functional::topk(const Tensor& x, int32_t k, int32_t dim = -1, bool largest = true, bool sorted = true)

   Find the top-k values and their indices along a given dimension.

   :param x: Input tensor
   :param k: Number of top elements to retrieve
   :param dim: Dimension along which to find top-k elements (default: -1, last dimension)
   :param largest: Whether to return largest (true) or smallest (false) elements (default: true)
   :param sorted: Whether to return elements in sorted order (default: true)
   :return: Array containing values tensor and indices tensor

Element-wise Operations
-----------------------

.. cpp:function:: Tensor mllm::nn::functional::clip(const Tensor& x, float min_val, float max_val)

   Clip (limit) the values in a tensor to a specified range.

   :param x: Input tensor
   :param min_val: Minimum value
   :param max_val: Maximum value
   :return: Clipped tensor

Reduction Operations
--------------------

.. cpp:function:: Tensor mllm::nn::functional::min(const Tensor& x, int32_t dim = std::numeric_limits<int32_t>::max(), bool keep_dim = false)

   Compute the minimum value of elements in the tensor.

   :param x: Input tensor
   :param dim: Dimension along which to compute minimum. If max int32_t, compute over all dimensions (default: max int32_t)
   :param keep_dim: Whether to keep the reduced dimension (default: false)
   :return: Tensor with minimum values

.. cpp:function:: Tensor mllm::nn::functional::max(const Tensor& x, int32_t dim = std::numeric_limits<int32_t>::max(), bool keep_dim = false)

   Compute the maximum value of elements in the tensor.

   :param x: Input tensor
   :param dim: Dimension along which to compute maximum. If max int32_t, compute over all dimensions (default: max int32_t)
   :param keep_dim: Whether to keep the reduced dimension (default: false)
   :return: Tensor with maximum values

.. cpp:function:: Tensor mllm::nn::functional::sum(const Tensor& x, int32_t dim = std::numeric_limits<int32_t>::max(), bool keep_dim = false)

   Compute the sum of elements in the tensor.

   :param x: Input tensor
   :param dim: Dimension along which to compute sum. If max int32_t, compute over all dimensions (default: max int32_t)
   :param keep_dim: Whether to keep the reduced dimension (default: false)
   :return: Tensor with sum values

.. cpp:function:: Tensor mllm::nn::functional::mean(const Tensor& x, int32_t dim = std::numeric_limits<int32_t>::max(), bool keep_dim = false)

   Compute the mean of elements in the tensor.

   :param x: Input tensor
   :param dim: Dimension along which to compute mean. If max int32_t, compute over all dimensions (default: max int32_t)
   :param keep_dim: Whether to keep the reduced dimension (default: false)
   :return: Tensor with mean values