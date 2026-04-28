// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/graph/AscendCausalMaskPluginOperation.hpp"

#include "mllm/backends/ascend/ops/AscendCausalMaskKernel.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::ascend {

namespace MLLM_ANONYMOUS_NAMESPACE {

constexpr uint32_t INPUT_NUM = 1;
constexpr uint32_t OUTPUT_NUM = 1;
constexpr uint32_t INPUT_INDEX = 0;
constexpr uint32_t OUTPUT_INDEX = 0;

}  // namespace MLLM_ANONYMOUS_NAMESPACE

AscendCausalMaskPluginOperation::AscendCausalMaskPluginOperation(bool sliding_window, int32_t window_size)
    : sliding_window_(sliding_window), window_size_(window_size) {}

std::string AscendCausalMaskPluginOperation::GetName() const {
  return "AscendCausalMaskPluginOperation";
}

atb::Status AscendCausalMaskPluginOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs,
                                                        atb::SVector<atb::TensorDesc>& outTensorDescs) const {
  if (inTensorDescs.size() != INPUT_NUM || outTensorDescs.size() != OUTPUT_NUM) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  outTensorDescs.at(OUTPUT_INDEX) = inTensorDescs.at(INPUT_INDEX);
  return atb::NO_ERROR;
}

uint32_t AscendCausalMaskPluginOperation::GetInputNum() const {
  return INPUT_NUM;
}

uint32_t AscendCausalMaskPluginOperation::GetOutputNum() const {
  return OUTPUT_NUM;
}

atb::Status AscendCausalMaskPluginOperation::Setup(const atb::VariantPack& variantPack,
                                                   uint64_t& workspace_size,
                                                   atb::Context* context) {
  (void)context;
  if (variantPack.inTensors.size() != INPUT_NUM || variantPack.outTensors.size() != OUTPUT_NUM) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  workspace_size = 0;
  return atb::NO_ERROR;
}

atb::Status AscendCausalMaskPluginOperation::Execute(const atb::VariantPack& variantPack,
                                                     uint8_t* workspace,
                                                     uint64_t workspace_size,
                                                     atb::Context* context) {
  (void)workspace;
  (void)workspace_size;
  (void)context;
  if (variantPack.inTensors.size() != INPUT_NUM || variantPack.outTensors.size() != OUTPUT_NUM) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  auto output = variantPack.outTensors.at(OUTPUT_INDEX);
  return executeAscendCausalMaskKernel(variantPack.inTensors.at(INPUT_INDEX), output, sliding_window_, window_size_);
}

atb::Operation* createCausalMaskPluginGraphOp(bool sliding_window, int32_t window_size) {
  return new AscendCausalMaskPluginOperation(sliding_window, window_size);
}

}  // namespace mllm::ascend
