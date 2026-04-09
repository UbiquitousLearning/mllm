// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/graph/AscendCausalMaskPluginOperation.hpp"

#include "mllm/backends/ascend/ops/AscendCausalMaskKernel.hpp"

namespace mllm::ascend {

AscendCausalMaskPluginOperation::AscendCausalMaskPluginOperation(bool sliding_window, int32_t window_size)
    : sliding_window_(sliding_window), window_size_(window_size) {}

std::string AscendCausalMaskPluginOperation::GetName() const {
  return "AscendCausalMaskPluginOperation";
}

atb::Status AscendCausalMaskPluginOperation::InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs,
                                                        atb::SVector<atb::TensorDesc>& outTensorDescs) const {
  if (inTensorDescs.size() != 1 || outTensorDescs.size() != 1) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  outTensorDescs.at(0) = inTensorDescs.at(0);
  return atb::NO_ERROR;
}

uint32_t AscendCausalMaskPluginOperation::GetInputNum() const {
  return 1;
}

uint32_t AscendCausalMaskPluginOperation::GetOutputNum() const {
  return 1;
}

atb::Status AscendCausalMaskPluginOperation::Setup(const atb::VariantPack& variantPack,
                                                   uint64_t& workspaceSize,
                                                   atb::Context* context) {
  (void)context;
  if (variantPack.inTensors.size() != 1 || variantPack.outTensors.size() != 1) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  workspaceSize = 0;
  return atb::NO_ERROR;
}

atb::Status AscendCausalMaskPluginOperation::Execute(const atb::VariantPack& variantPack,
                                                     uint8_t* workspace,
                                                     uint64_t workspaceSize,
                                                     atb::Context* context) {
  (void)workspace;
  (void)workspaceSize;
  (void)context;
  if (variantPack.inTensors.size() != 1 || variantPack.outTensors.size() != 1) {
    return atb::ERROR_INVALID_TENSOR_NUM;
  }
  auto output = variantPack.outTensors.at(0);
  return executeAscendCausalMaskKernel(variantPack.inTensors.at(0), output, sliding_window_, window_size_);
}

atb::Operation* createCausalMaskPluginGraphOp(bool sliding_window, int32_t window_size) {
  return new AscendCausalMaskPluginOperation(sliding_window, window_size);
}

}  // namespace mllm::ascend
