// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "mllm/engine/Dispatcher.hpp"

namespace mllm::cpu {

struct CPUDispatcherOptions {
  int32_t queue_depth_ = 0;
  bool need_async_exec_ = true;
};

class CPUDispatcher final : public Dispatcher {
 public:
  using ptr_t = std::shared_ptr<CPUDispatcher>;

  explicit CPUDispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id, const CPUDispatcherOptions& options);

  void receive(const Task::ptr_t& task) override;

  void process(const Task::ptr_t& task) override;

  void syncWait() override;

 private:
  CPUDispatcherOptions options_;
};

CPUDispatcher::ptr_t createCPUDispatcher(exec::static_thread_pool& thread_pool, const CPUDispatcherOptions& options);

}  // namespace mllm::cpu
