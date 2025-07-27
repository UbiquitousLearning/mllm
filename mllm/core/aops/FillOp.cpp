/**
 * @file FillOp.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-26
 *
 */
#include "mllm/core/aops/FillOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::aops {

FillOp::FillOp(const FillOpOptions& options) : BaseOp(OpTypes::kFill), options_(options) {}

void FillOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void FillOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("FillOp::trace not implemented");
}

void FillOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("FillOp::forward not implemented in aops base.");
}

void FillOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { outputs.emplace_back(inputs[0]); }

void FillOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // The fillop is performed inplace
  // There is no need to alloc output again!
}

}  // namespace mllm::aops