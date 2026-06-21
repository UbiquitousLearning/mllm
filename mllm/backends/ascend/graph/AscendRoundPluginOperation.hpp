// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>

#include <atb/operation_infra.h>

namespace mllm::ascend {

// Lightweight ATB plugin that wraps aclnnRound (FP16 → FP16, banker's rounding).
// 1-in / 1-out; output shape/dtype matches input.
//
// Executor caching: when src/dst device pointers are unchanged between calls
// (steady-state decode in graph mode), GetWorkspaceSize is skipped entirely.
class AscendRoundPluginOperation final : public atb::OperationInfra {
 public:
  AscendRoundPluginOperation() = default;
  ~AscendRoundPluginOperation() override;

  std::string GetName() const override;
  atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs,
                         atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
  uint32_t GetInputNum() const override;
  uint32_t GetOutputNum() const override;
  atb::Status Setup(const atb::VariantPack& variantPack,
                    uint64_t& workspace_size,
                    atb::Context* context) override;
  atb::Status Execute(const atb::VariantPack& variantPack,
                      uint8_t* workspace,
                      uint64_t workspace_size,
                      atb::Context* context) override;

 private:
  // Opaque cache state that owns aclTensor/aclOpExecutor and their metadata.
  // Kept as void* to avoid pulling ACL headers into the public header.
  void* cache_state_{nullptr};
};

// Factory helper for use inside AscendGraphBuilder.
atb::Operation* createRoundPluginGraphOp();

}  // namespace mllm::ascend
