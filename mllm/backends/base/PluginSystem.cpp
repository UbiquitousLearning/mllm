// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/engine/Context.hpp"
#include "mllm/backends/base/PluginSystem.hpp"

namespace mllm::plugin {

int32_t OpPluginSystem::registerCustomizedOp(DeviceTypes device_type, const std::shared_ptr<BaseOpFactory>& factory) {
  auto op_type_ret = ++dynamic_op_type_counter_;
  Context::instance().getBackend(device_type)->regOpFactory(factory);
  factory->__forceSetType(op_type_ret);
  return op_type_ret;
}

}  // namespace mllm::plugin
