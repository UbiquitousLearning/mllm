/**
 * @file DispatcherManager.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-25
 *
 */
#pragma once

#include <cstdint>
#include <memory>

#include "mllm/engine/Task.hpp"
#include "mllm/engine/Dispatcher.hpp"
#include "mllm/utils/SymbolTable.hpp"

// C++26's Feature. Added by involving NVIDIA's stdexec
#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>

namespace mllm {

struct DispatcherManagerOptions {
  bool numa_policy = false;
  uint32_t num_threads = 0;
};

class DispatcherManager {
 public:
  using ptr_t = std::shared_ptr<DispatcherManager>;
  using dispatcher_id_t = int32_t;

  static constexpr int32_t cpu_dispatcher_id = Dispatcher::cpu_dispatcher_id;
  static constexpr int32_t cuda_dispatcher_id = Dispatcher::cuda_dispatcher_id;
  static constexpr int32_t opencl_dispatcher_id = Dispatcher::opencl_dispatcher_id;
  static constexpr int32_t qnn_dispatcher_id = Dispatcher::qnn_dispatcher_id;
  static constexpr int32_t custom_dispatcher_start_id = Dispatcher::qnn_dispatcher_id;

  explicit DispatcherManager(const DispatcherManagerOptions& options);

  inline exec::static_thread_pool& getExecutor() { return thread_pool_; }

  void submit(dispatcher_id_t id, const Task::ptr_t& task);

  void syncWait(dispatcher_id_t id);

  void registerDispatcher(const Dispatcher::ptr_t& dispatcher);

 private:
  DispatcherManagerOptions options_;
  exec::static_thread_pool thread_pool_;
  SymbolTable<int32_t, Dispatcher::ptr_t> dispatchers_;
};

}  // namespace mllm
