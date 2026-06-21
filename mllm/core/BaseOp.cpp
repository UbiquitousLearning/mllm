// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/BaseOp.hpp"

namespace mllm {

BaseOp::BaseOp(OpTypes op_type) : op_type_(op_type) {}

void BaseOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  for (auto& t : outputs) {
    if (!t.isNil()) t.alloc();
  }
}

std::string BaseOp::getName() const { return name_; }

void BaseOp::setName(const std::string& name) { name_ = name; }

DeviceTypes BaseOp::getDevice() const { return device_type_; }

void BaseOp::setDeviceType(DeviceTypes device_type) { device_type_ = device_type; }

OpTypes BaseOp::getOpType() const { return op_type_; }

void BaseOp::setOpType(OpTypes op_type) { op_type_ = op_type; }

}  // namespace mllm
