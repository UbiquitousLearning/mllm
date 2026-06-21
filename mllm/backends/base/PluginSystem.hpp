// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

// Plugin system for:
// 1. Plugin your own backends
// 2. Plugin your own operators into a existing backend

#include <string>

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/utils/SymbolTable.hpp"
#include "mllm/backends/base/PluginInterface.hpp"

namespace mllm::plugin {

class OpPluginSystem {
 public:
  using op_type_t = int32_t;
  using op_name_t = std::string;

  OpPluginSystem() = default;

  ~OpPluginSystem();

  void loadOpPackage(const std::string& path);

  int32_t registerCustomizedOp(DeviceTypes device_type, const std::string& name, const std::shared_ptr<BaseOpFactory>& factory);

  int32_t lookupCustomizedOp(DeviceTypes device_type, const std::string& name);

 private:
  int32_t dynamic_op_type_counter_ = (int32_t)OpTypes::kDynamicOp_Start;
  SymbolTable<DeviceTypes, SymbolTable<op_name_t, op_type_t>> op_name_table_;
  std::vector<PluginOpPackageDescriptor*> op_packages_;
#ifdef _WIN32
  std::vector<HMODULE> loaded_libraries_;
#else
  std::vector<void*> loaded_libraries_;
#endif
};

}  // namespace mllm::plugin
