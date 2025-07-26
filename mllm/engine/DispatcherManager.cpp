/**
 * @file DispatcherManager.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-25
 *
 */
#include "mllm/engine/DispatcherManager.hpp"
#include "exec/static_thread_pool.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm {

DispatcherManager::DispatcherManager(const DispatcherManagerOptions& options)
    : options_(options), thread_pool_(options.num_threads) {
  if (options.numa_policy) { MLLM_WARN("NUMA policy is not supported yet."); }
  exec::numa_policy numa{exec::no_numa_policy{}};
}

void DispatcherManager::submit(dispatcher_id_t id, const Task::ptr_t& task) { dispatchers_[id]->receive(task); }

void DispatcherManager::syncWait(dispatcher_id_t id) { dispatchers_[id]->syncWait(); }

void DispatcherManager::registerDispatcher(const Dispatcher::ptr_t& dispatcher) {
  dispatchers_.reg(dispatcher->id(), dispatcher);
}

}  // namespace mllm
