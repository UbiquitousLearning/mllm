// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/ViewOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

ViewOp::ViewOp(const ViewOpOptions& options) : BaseOp(OpTypes::kView), options_(options) {}

void ViewOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void ViewOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::ViewOp>(shared_from_this(), i_irs, o_irs);
}

void ViewOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

void ViewOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& it = inputs[0];
  auto const& new_shape = options_.to_shape;

  std::vector<int32_t> actual_shape = new_shape;
  int infer_dim = -1;
  size_t product = 1;

  for (size_t i = 0; i < actual_shape.size(); ++i) {
    if (actual_shape[i] == -1) {
      // only one dimension can be inferred
      MLLM_RT_ASSERT(infer_dim == -1);
      infer_dim = static_cast<int>(i);
    } else {
      product *= actual_shape[i];
    }
  }

  // infer dim
  if (infer_dim != -1) {
    size_t input_numel = it.numel();
    MLLM_RT_ASSERT(product != 0);
    MLLM_RT_ASSERT(input_numel % product == 0);
    actual_shape[infer_dim] = static_cast<int32_t>(input_numel / product);
  }

  // check numel
  size_t new_numel = 1;
  for (int dim : actual_shape) { new_numel *= dim; }
  MLLM_RT_ASSERT_EQ(it.numel(), new_numel);

  auto orig_storage = it.impl()->storage();
  int32_t orig_storage_offset = it.impl()->storageOffset();
  auto orig_stride = it.impl()->stride();
  auto orig_shape = it.impl()->shape();

  std::vector<int32_t> new_stride(actual_shape.size(), 0);
  int orig_dim = orig_shape.size() - 1;
  int64_t right_product = 1;

  for (int new_dim = actual_shape.size() - 1; new_dim >= 0; --new_dim) {
    int64_t size = actual_shape[new_dim];
    if (orig_dim >= 0 && (orig_shape[orig_dim] == 1 || right_product * size <= orig_shape[orig_dim])) {
      right_product *= size;
      if (right_product == orig_shape[orig_dim]) {
        new_stride[new_dim] = orig_stride[orig_dim] * (right_product / size);
        right_product = 1;
        orig_dim--;
      } else {
        new_stride[new_dim] = orig_stride[orig_dim] * (right_product / size);
      }
    } else {
      if (new_dim < actual_shape.size() - 1) {
        new_stride[new_dim] = actual_shape[new_dim + 1] * new_stride[new_dim + 1];
      } else {
        new_stride[new_dim] = 1;
      }
    }
  }

  for (; orig_dim >= 0; --orig_dim) {
    if (actual_shape[0] == 1) { new_stride[0] = orig_stride[orig_dim]; }
  }

  outputs.emplace_back(TensorViewImpl::create(orig_storage_offset, actual_shape, new_stride, orig_storage));
}

void ViewOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_EMPTY_SCOPE;
  // DO nothing.
}

}  // namespace mllm::aops