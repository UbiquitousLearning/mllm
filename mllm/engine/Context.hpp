/**
 * @file Context.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-21
 *
 */
#pragma once

#include "mllm/core/DeviceTypes.hpp"
#include "mllm/utils/SymbolTable.hpp"
#include "mllm/engine/MemoryManager.hpp"
#include "mllm/backends/base/Backend.hpp"

namespace mllm {

class Context {
 public:
  static Context& instance() {
    static Context instance;
    return instance;
  }

  Context();

  Context(const Context&) = delete;

  Context& operator=(const Context&) = delete;

  inline Backend::ptr_t getBackend(const DeviceTypes& device) {
    if (!backends_.has(device)) {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Backend for device {} not found", deviceTypes2Str(device));
    }
    return backends_[device];
  }

  inline MemoryManager::ptr_t memoryManager() { return memory_manager_; }

 private:
  MemoryManager::ptr_t memory_manager_ = nullptr;
  SymbolTable<DeviceTypes, Backend::ptr_t> backends_;
};

}  // namespace mllm