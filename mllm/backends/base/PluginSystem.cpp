// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <string>
#include <stdexcept>

#include "mllm/engine/Context.hpp"
#include "mllm/backends/base/PluginSystem.hpp"

namespace mllm::plugin {

OpPluginSystem::~OpPluginSystem() {
#ifdef _WIN32
  for (auto handle : loaded_libraries_) {
    if (handle) { FreeLibrary(handle); }
  }
#else
  for (auto handle : loaded_libraries_) {
    if (handle) { dlclose(handle); }
  }
#endif
  for (auto descriptor : op_packages_) {
    if (descriptor) { delete descriptor; }
  }
}

void OpPluginSystem::loadOpPackage(const std::string& path) {
  // 1. DLOPEN
#ifdef _WIN32
  HMODULE handle = LoadLibraryA(path.c_str());
  if (!handle) { throw std::runtime_error("Failed to load plugin library on Windows: " + path); }
#else  // Linux and other Unix-like systems
  void* handle = dlopen(path.c_str(), RTLD_LAZY);
  if (!handle) {
    throw std::runtime_error("Failed to load plugin library on Linux and other Unix-like systems: " + std::string(dlerror()));
  }
#endif

  // 2. Call function opPackageDescriptor() in the library
  typedef void* (*DescriptorFunc)();  // NOLINT
#ifdef _WIN32
  DescriptorFunc descriptor = (DescriptorFunc)GetProcAddress(handle, "opPackageDescriptor");
  if (!descriptor) {
    FreeLibrary(handle);
    throw std::runtime_error("Failed to find opPackageDescriptor function in plugin: " + path);
  }
#else
  DescriptorFunc descriptor = (DescriptorFunc)dlsym(handle, "opPackageDescriptor");
  if (!descriptor) {
    dlclose(handle);
    throw std::runtime_error("Failed to find opPackageDescriptor function in plugin: " + std::string(dlerror()));
  }
#endif

  // 3. Cast void* return to PluginOpPackageDescriptor*
  void* result = descriptor();
  PluginOpPackageDescriptor* package_descriptor = static_cast<PluginOpPackageDescriptor*>(result);
  MLLM_RT_ASSERT(package_descriptor != nullptr);
  op_packages_.push_back(package_descriptor);
  loaded_libraries_.push_back(handle);

  // 4. Load all factory to the backend
  MLLM_INFO("Load customized op package: {}, find op nums: {}", package_descriptor->name,
            package_descriptor->op_factories_count);
  for (int i = 0; i < package_descriptor->op_factories_count; ++i) {
    std::shared_ptr<BaseOpFactory> factory((BaseOpFactory*)(package_descriptor->op_factory_create_funcs[i]()),
                                           [&package_descriptor, &i](BaseOpFactory* ptr) {
                                             // FIXME: will raise error
                                             // package_descriptor->op_factory_free_funcs[i](ptr);
                                           });
    registerCustomizedOp(static_cast<DeviceTypes>(package_descriptor->device_type),
                         std::string(package_descriptor->op_factories_names[i]), factory);
  }
}

int32_t OpPluginSystem::registerCustomizedOp(DeviceTypes device_type, const std::string& name,
                                             const std::shared_ptr<BaseOpFactory>& factory) {
  auto op_type_ret = ++dynamic_op_type_counter_;
  factory->__forceSetType(op_type_ret);
  Context::instance().getBackend(device_type)->regOpFactory(factory);
  if (!op_name_table_.has(device_type)) { op_name_table_.reg(device_type, SymbolTable<op_name_t, op_type_t>{}); }
  op_name_table_[device_type].reg(name, op_type_ret);
  MLLM_INFO("Register customized op: {}:{} -> {}", name, op_type_ret, deviceTypes2Str(device_type));
  return op_type_ret;
}

int32_t OpPluginSystem::lookupCustomizedOp(DeviceTypes device_type, const std::string& name) {
  return op_name_table_[device_type][name];
}

}  // namespace mllm::plugin
