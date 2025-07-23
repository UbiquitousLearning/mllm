/**
 * @file Backend.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#include "mllm/backends/base/Backend.hpp"

namespace mllm {

Backend::Backend(DeviceTypes device, const Allocator::ptr_t& allocator) : device_(device), allocator_(allocator) {}

BaseOp::ptr_t Backend::createOp(OpTypes op_type, const BaseOpOptionsBase& base_options) {
  auto op = op_factories_[op_type]->create(base_options);
  op->setDeviceType(device_);
  return op;
}

}  // namespace mllm
