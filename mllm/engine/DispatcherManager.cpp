// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/engine/DispatcherManager.hpp"
#include "exec/static_thread_pool.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/tracy_perf/Tracy.hpp"

namespace mllm {

DispatcherManager::DispatcherManager(const DispatcherManagerOptions& options)
    : options_(options), thread_pool_(options.num_threads) {
  if (options.numa_policy) { MLLM_WARN("NUMA policy is not supported yet."); }
  exec::numa_policy numa{exec::no_numa_policy{}};
}

void DispatcherManager::submit(dispatcher_id_t id, const Task::ptr_t& task) {
  MLLM_TRACY_ZONE_SCOPED;
  dispatchers_[id]->receive(task);
}

TaskResult::sender_t DispatcherManager::asyncSubmit(dispatcher_id_t id, const Task::ptr_t& task) {
  return dispatchers_[id]->asyncReceive(task);
}

void DispatcherManager::syncWait(dispatcher_id_t id) { dispatchers_[id]->syncWait(); }

void DispatcherManager::registerDispatcher(const Dispatcher::ptr_t& dispatcher) {
  dispatchers_.reg(dispatcher->id(), dispatcher);
}

bool DispatcherManager::hasDispatcher(dispatcher_id_t id) { return dispatchers_.has(id); }

Dispatcher::ptr_t DispatcherManager::getDispatcher(dispatcher_id_t id) { return dispatchers_[id]; }
}  // namespace mllm
