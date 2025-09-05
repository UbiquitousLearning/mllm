// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/SliceOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

SliceOp::SliceOp(const SliceOpOptions& options) : BaseOp(OpTypes::kSlice), options_(options) {}

void SliceOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void SliceOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::SliceOp>(shared_from_this(), i_irs, o_irs);
}

void SliceOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

void SliceOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];

  if (!input.impl()) { outputs.emplace_back(Tensor::nil()); }

  auto shape = input.shape();
  auto slice_index = options_.indices_;

  MLLM_RT_ASSERT_EQ(slice_index.size(), shape.size());

  auto old_impl = input.impl();
  auto old_storage = old_impl->storage();
  int32_t old_rank = shape.size();
  std::vector<int32_t> new_shape;
  int32_t new_storage_offset = old_impl->storageOffset();
  std::vector<int32_t> new_stride;

  for (int i = 0; i < old_rank; ++i) {
    const auto& pair = slice_index[i];
    int32_t start = pair.start_;
    int32_t end = pair.end_;
    int32_t step = pair.step_;

    if (start == kAll) { start = 0; }
    if (end == kAll) { end = shape[i]; }

    if (start < 0 && end != kAll && end - start == 1) {
      start = start + shape[i];
      end = end + shape[i];
    }
    if (start < 0) { start = start + shape[i]; }
    if (end < 0) { end = end + shape[i]; }

    if (step < 1) { NYI("Mllm only support step >= 1 in operator[] right now"); }

    int32_t num_elements = 0;
    if (end > start) { num_elements = (end - start + step - 1) / step; }

    new_storage_offset += start * old_impl->stride()[i];
    new_stride.push_back(old_impl->stride()[i] * step);
    new_shape.push_back(num_elements);
  }

  auto new_impl = TensorViewImpl::create(new_storage_offset, new_shape, new_stride, old_storage);
  outputs.emplace_back(new_impl);
}

void SliceOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

}  // namespace mllm::aops
