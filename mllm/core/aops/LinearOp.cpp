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
#include "mllm/utils/Common.hpp"

namespace mllm::aops {

LinearOp::LinearOp(const LinearOptions& options) : BaseOp(OpTypes::kLinear), options_(options) {}

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
  // TODO
}

void LinearOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // TODO
}

ParameterFile::ptr_t LinearOp::getParams() {
  auto p = ParameterFile::create();
  p->push(getName() + ".weight", weight_);
  if (options_.bias) { p->push(getName() + ".bias", bias_); }
  return p;
}

}  // namespace mllm::aops