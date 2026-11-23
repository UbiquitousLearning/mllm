// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/LinearShadowCorrectionOp.hpp"

namespace mllm::cpu {

class CPULinearShadowCorrectionOp final : public aops::LinearShadowCorrectionOp {
 public:
  explicit CPULinearShadowCorrectionOp(const aops::LinearShadowCorrectionOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  void forwardOriginal(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);
  void forwardPacket(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);
};

class CPULinearShadowCorrectionOpFactory
    : public TypedOpFactory<OpTypes::kLinearShadowCorrection, aops::LinearShadowCorrectionOpOptions> {
 protected:
  std::shared_ptr<BaseOp> createOpImpl(const aops::LinearShadowCorrectionOpOptions& options) override {
    return std::make_shared<CPULinearShadowCorrectionOp>(options);
  }
};

}  // namespace mllm::cpu
