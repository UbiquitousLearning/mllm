// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/SplitOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

SplitOp::SplitOp(const SplitOpOptions& options) : BaseOp(OpTypes::kSplit), options_(options) {}

void SplitOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void SplitOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::SplitOp>(shared_from_this(), i_irs, o_irs);
}

void SplitOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

void SplitOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto const& it = inputs[0];
  int split_at_dim = options_.dim;
  std::vector<int> section_sizes;

  if (split_at_dim < 0) split_at_dim = it.shape().size() + split_at_dim;

  if (options_.split_size_or_sections.size() == 1) {
    MLLM_RT_ASSERT_EQ(it.shape()[split_at_dim] % options_.split_size_or_sections[0], 0);

    for (int i = 0; i < it.shape()[split_at_dim] / options_.split_size_or_sections[0]; ++i) {
      section_sizes.push_back(options_.split_size_or_sections[0]);
    }
  } else {
    int cnt = 0;
    for (int split_size_or_section : options_.split_size_or_sections) { cnt += split_size_or_section; }

    MLLM_RT_ASSERT_EQ(cnt, it.shape()[split_at_dim]);

    for (int split_size_or_section : options_.split_size_or_sections) { section_sizes.push_back(split_size_or_section); }
  }

  // Ok. We can now start to split the tensor. Pls calculate storage offsets and stride carefully.
  auto orig_storage = it.impl()->storage();
  int32_t orig_storage_offset = it.impl()->storageOffset();
  auto orig_stride = it.impl()->stride();
  auto orig_shape = it.impl()->shape();

  int sum = 0;
  for (int section_size : section_sizes) {
    std::vector<int32_t> new_shape(orig_shape.begin(), orig_shape.end());
    new_shape[split_at_dim] = section_size;

    int32_t new_storage_offset = orig_storage_offset + sum * orig_stride[split_at_dim];

    outputs.emplace_back(TensorViewImpl::create(new_storage_offset, new_shape, orig_stride, orig_storage));

    sum += section_size;
  }
}

void SplitOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Do nothing
  MLLM_EMPTY_SCOPE;
}

}  // namespace mllm::aops