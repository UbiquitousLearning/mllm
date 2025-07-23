/**
 * @file Context.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-21
 *
 */
#pragma once

#include <atomic>

#include "mllm/core/DeviceTypes.hpp"
#include "mllm/engine/SessionTCB.hpp"
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

  void registerBackend(const Backend::ptr_t& new_backend);

  Backend::ptr_t getBackend(const DeviceTypes& device);

  inline MemoryManager::ptr_t memoryManager() { return memory_manager_; }

  uint32_t getUUID();

  SessionTCB::ptr_t thisThread();

  SessionTCB::ptr_t mainThread();

 private:
  std::shared_ptr<SessionTCB> main_thread_;
  std::unordered_map<std::thread::id, SessionTCB::ptr_t> session_threads_;

  std::atomic<uint32_t> custom_uuid_giver_ = 0;
  MemoryManager::ptr_t memory_manager_ = nullptr;
  SymbolTable<DeviceTypes, Backend::ptr_t> backends_;
};

}  // namespace mllm