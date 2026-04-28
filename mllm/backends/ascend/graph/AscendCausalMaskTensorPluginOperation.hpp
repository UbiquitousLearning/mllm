// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>

#include <atb/operation_infra.h>

namespace mllm::ascend {

// Builds an additive causal mask tensor for KV-cache attention.
// Input 0 provides the target shape/dtype; input 1 is current_seq_len on device.
class AscendCausalMaskTensorPluginOperation final : public atb::OperationInfra {
 public:
  AscendCausalMaskTensorPluginOperation(bool sliding_window = false, int32_t window_size = 0);
  ~AscendCausalMaskTensorPluginOperation() override;

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
  void* host_mask_buffer_{nullptr};
  uint64_t host_mask_buffer_bytes_{0};

  atb::Status ensureHostMaskBuffer(uint64_t required_bytes);
};

atb::Operation* createCausalMaskTensorPluginGraphOp(bool sliding_window = false, int32_t window_size = 0);

}  // namespace mllm::ascend
