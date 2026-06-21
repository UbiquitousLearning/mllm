// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/experiments/auto_tune/TuningSpace.hpp"

//===----------------------------------------------------------------------===//
// Elementwise Add
//===----------------------------------------------------------------------===//
class ElewiseAddFloat32 : public mllm::OpTunningSpace {
 public:
  ElewiseAddFloat32();

  std::vector<mllm::Tensor> buildInputs(const std::unordered_map<std::string, std::any>& space) override;

  mllm::BaseOp::ptr_t buildOp(const std::unordered_map<std::string, std::any>& space) override;

  void writeBackTunningSpace(const std::unordered_map<std::string, std::any>& space) override;
};
