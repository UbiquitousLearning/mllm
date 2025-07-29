/**
 * @file BaseOp.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#include "mllm/core/BaseOp.hpp"

namespace mllm {

BaseOp::BaseOp(OpTypes op_type) : op_type_(op_type) {}

void BaseOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  for (auto& t : outputs) { t.alloc(); }
}

std::string BaseOp::getName() const { return name_; }

void BaseOp::setName(const std::string& name) { name_ = name; }

DeviceTypes BaseOp::getDevice() const { return device_type_; }

void BaseOp::setDeviceType(DeviceTypes device_type) { device_type_ = device_type; }

OpTypes BaseOp::getOpType() const { return op_type_; }

}  // namespace mllm
