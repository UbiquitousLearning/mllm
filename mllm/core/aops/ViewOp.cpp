// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/ViewOp.hpp"
#include <numeric>
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::aops {

ViewOp::ViewOp(const ViewOpOptions& options) : BaseOp(OpTypes::kView), options_(options) {}

void ViewOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void ViewOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs, true);  // no_memory_side_effect=True
  ir_ctx->create<ir::linalg::ViewOp>(shared_from_this(), i_irs, o_irs);
}

void ViewOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

void ViewOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& it = inputs[0];
  auto const& new_shape = options_.to_shape;

  // check shape and element num
  std::vector<int32_t> actual_shape = new_shape;
  int infer_dim = -1;
  size_t product = 1;

  for (size_t i = 0; i < actual_shape.size(); ++i) {
    if (actual_shape[i] == -1) {
      MLLM_RT_ASSERT(infer_dim == -1 && "only one dimension can be inferred");
      infer_dim = static_cast<int>(i);
    } else {
      // check non-negative
      MLLM_RT_ASSERT(actual_shape[i] >= 0 && "shape dimensions must be non-negative");
      product *= actual_shape[i];
    }
  }

  // infer dim
  const int64_t input_numel = it.numel();
  if (infer_dim != -1) {
    MLLM_RT_ASSERT(product != 0 && "cannot infer dimension for a shape with zero product");
    MLLM_RT_ASSERT(input_numel % product == 0 && "input tensor size does not match inferred shape");
    actual_shape[infer_dim] = static_cast<int32_t>(input_numel / product);
  }

  const int64_t new_numel = std::accumulate(actual_shape.begin(), actual_shape.end(), 1, std::multiplies<>());
  MLLM_RT_ASSERT_EQ(input_numel, new_numel);

  auto orig_storage = it.impl()->storage();
  int32_t orig_storage_offset = it.impl()->storageOffset();
  const auto& orig_shape = it.impl()->shape();
  const auto& orig_stride = it.impl()->stride();

  // if shape is same, return itself
  if (orig_shape == actual_shape) {
    outputs.emplace_back(TensorViewImpl::create(orig_storage_offset, orig_shape, orig_stride, orig_storage));
    return;
  }

  // Check if the tensor can be Viewed

  bool is_contiguous = true;
  int64_t current_stride = 1;
  for (int i = orig_shape.size() - 1; i >= 0; --i) {
    if (orig_stride[i] != current_stride) {
      is_contiguous = false;
      break;
    }
    // size == 1 does not affect contiguity
    if (orig_shape[i] != 1) { current_stride *= orig_shape[i]; }
  }

  std::vector<int32_t> new_stride(actual_shape.size());

  if (is_contiguous) {
    current_stride = 1;
    for (int i = actual_shape.size() - 1; i >= 0; --i) {
      new_stride[i] = static_cast<int32_t>(current_stride);
      current_stride *= actual_shape[i];
    }
  } else {
    // FIXME: more stride logic such as `compute_stride_for_view` in PyTorch
    MLLM_ASSERT_EXIT(is_contiguous, "ViewOp::reshape is only supported for contiguous tensors in this implementation");
  }

  outputs.emplace_back(TensorViewImpl::create(orig_storage_offset, actual_shape, new_stride, orig_storage));
}

void ViewOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_EMPTY_SCOPE;
  // DO nothing.
}

}  // namespace mllm::aops
