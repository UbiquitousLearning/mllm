// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/backends/base/Backend.hpp"

namespace mllm {

Backend::Backend(DeviceTypes device, const Allocator::ptr_t& allocator) : device_(device), allocator_(allocator) {}

BaseOp::ptr_t Backend::createOp(OpTypes op_type, const BaseOpOptionsBase& base_options) {
  auto op = op_factories_[op_type]->create(base_options);
  op->setDeviceType(device_);
  return op;
}

void Backend::regOpFactory(const std::shared_ptr<BaseOpFactory>& factory) { op_factories_.reg(factory->opType(), factory); }

}  // namespace mllm
