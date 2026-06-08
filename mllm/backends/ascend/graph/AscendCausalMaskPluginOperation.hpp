// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>

#include <atb/operation_infra.h>

namespace mllm::ascend {

// Standalone graph wrapper for applying a causal mask to an existing scores tensor.
// KV-cache attention uses AscendCausalMaskTensorPluginOperation instead, because it
// needs current_seq_len and an additive mask tensor.
class AscendCausalMaskPluginOperation final : public atb::OperationInfra {
 public:
  AscendCausalMaskPluginOperation(bool sliding_window = false, int32_t window_size = 0);
  ~AscendCausalMaskPluginOperation() override = default;

  std::string GetName() const override;
  atb::Status InferShape(const atb::SVector<atb::TensorDesc>& inTensorDescs,
                         atb::SVector<atb::TensorDesc>& outTensorDescs) const override;
  uint32_t GetInputNum() const override;
  uint32_t GetOutputNum() const override;
  atb::Status Setup(const atb::VariantPack& variantPack, uint64_t& workspace_size, atb::Context* context) override;
  atb::Status Execute(const atb::VariantPack& variantPack,
                      uint8_t* workspace,
                      uint64_t workspace_size,
                      atb::Context* context) override;

 private:
  bool sliding_window_;
  int32_t window_size_;
};

atb::Operation* createCausalMaskPluginGraphOp(bool sliding_window = false, int32_t window_size = 0);

}  // namespace mllm::ascend
