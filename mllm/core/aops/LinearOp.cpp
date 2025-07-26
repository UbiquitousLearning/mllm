/**
 * @file LinearOp.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#include "mllm/core/aops/LinearOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::aops {

LinearOp::LinearOp(const LinearOpOptions& options) : BaseOp(OpTypes::kLinear), options_(options) {}

void LinearOp::load(const ParameterFile::ptr_t& ploader) {
  switch (ploader->version()) {
    case ModelFileVersion::kV1: {
      weight_ = ploader->pull(getName() + ".weight");
      if (options_.bias) { bias_ = ploader->pull(getName() + ".bias"); }
      // TODO Need to reshape
      break;
    }
    case ModelFileVersion::kUserTemporary:
    case ModelFileVersion::kV2: {
      weight_ = ploader->pull(getName() + ".weight");
      if (options_.bias) { bias_ = ploader->pull(getName() + ".bias"); }
      break;
    }
    default: NYI("Unsupported model file version")
  }
}

void LinearOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // TODO
}

void LinearOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("LinearOp::forward not implemented in aops base.");
}

void LinearOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto out_shape = input.shape();
  MLLM_RT_ASSERT_EQ(out_shape[out_shape.size() - 2], options_.in_channels);
  out_shape[out_shape.size() - 1] = options_.out_channels;

  outputs.emplace_back(Tensor::empty(out_shape, input.dtype(), input.device()));
}

void LinearOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

ParameterFile::ptr_t LinearOp::getParams() {
  auto p = ParameterFile::create();
  p->push(getName() + ".weight", weight_);
  if (options_.bias) { p->push(getName() + ".bias", bias_); }
  return p;
}

}  // namespace mllm::aops