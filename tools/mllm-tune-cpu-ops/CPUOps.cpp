// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/DeviceTypes.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/core/aops/ElewiseOps.hpp"
#include "mllm/engine/Context.hpp"

#include "CPUOps.hpp"

//===----------------------------------------------------------------------===//
// Elementwise Add
//===----------------------------------------------------------------------===//
ElewiseAddFloat32::ElewiseAddFloat32() : mllm::OpTunningSpace(mllm::OpTypes::kAdd, mllm::kCPU) {}

std::vector<mllm::Tensor> ElewiseAddFloat32::buildInputs(const std::unordered_map<std::string, std::any>& space) {
  auto size = std::any_cast<int32_t>(space.at("size"));
  return {
      mllm::Tensor::empty({size, size}, mllm::kFloat32, mllm::kCPU).alloc(),
      mllm::Tensor::empty({size, size}, mllm::kFloat32, mllm::kCPU).alloc(),
  };
}

mllm::BaseOp::ptr_t ElewiseAddFloat32::buildOp(const std::unordered_map<std::string, std::any>& space) {
  auto threads = std::any_cast<int32_t>(space.at("threads"));

  mllm::aops::AddOpOptions options;
  options.setThreads(threads);

  return mllm::Context::instance().getBackend(device_type_)->createOp(op_type_, options);
}

void ElewiseAddFloat32::writeBackTunningSpace(const std::unordered_map<std::string, std::any>& space) {
  config_.data()["ElewiseAddFloat32"]["threads"] = std::any_cast<int32_t>(space.at("threads"));
}
