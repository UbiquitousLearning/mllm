// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>

#include <atb/operation_infra.h>

namespace mllm::ascend {

// Lightweight ATB plugin that wraps aclnnClamp (FP16 → FP16).
// 1-in / 1-out; output shape/dtype matches input.
// Default range [-128, 127] is suitable for INT8 activation quantization.
class AscendClampPluginOperation final : public atb::OperationInfra {
 public:
  explicit AscendClampPluginOperation(float min_val = -128.f, float max_val = 127.f);
  ~AscendClampPluginOperation() override;

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
  float min_val_;
  float max_val_;

  // Opaque cache state that owns aclTensor/aclScalar/aclOpExecutor.
  void* cache_state_{nullptr};
};

// Factory helper for use inside AscendGraphBuilder.
atb::Operation* createClampPluginGraphOp(float min_val = -128.f, float max_val = 127.f);

}  // namespace mllm::ascend
