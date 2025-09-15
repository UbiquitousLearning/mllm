// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <any>
#include <vector>
#include <string>
#include <unordered_map>

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/engine/ConfigFile.hpp"

namespace mllm {

class OpTunningSpace {
 public:
  OpTunningSpace(OpTypes op_type, DeviceTypes device_type);

  void addTuningParameter(const std::string& name, const std::vector<std::any>& values);

  virtual BaseOp::ptr_t buildOp(const std::unordered_map<std::string, std::any>& space) = 0;

  virtual std::vector<Tensor> buildInputs(const std::unordered_map<std::string, std::any>& space) = 0;

  virtual void writeBackTunningSpace(const std::unordered_map<std::string, std::any>& space) = 0;

  virtual void tune();

 protected:
  [[nodiscard]] std::vector<std::unordered_map<std::string, std::any>> product() const;

  OpTypes op_type_;
  DeviceTypes device_type_;
  ConfigFile config_;

  std::unordered_map<std::string, std::vector<std::any>> space_;
};

}  // namespace mllm
