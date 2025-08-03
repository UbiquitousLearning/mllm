// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "mllm/engine/Dispatcher.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::ir {

struct IRTraceDispatcherOptions {
  MLLM_EMPTY_SCOPE;
};

class IRTraceDispatcher final : public Dispatcher {
 public:
  using ptr_t = std::shared_ptr<IRTraceDispatcher>;

  explicit IRTraceDispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id,
                             const IRTraceDispatcherOptions& options);

  void preprocessTask(const Task::ptr_t& task) override;

  void receive(const Task::ptr_t& task) override;

  TaskResult::sender_t asyncReceive(const Task::ptr_t& task) override;

  void process(const Task::ptr_t& task) override;

  void syncWait() override;

 private:
  IRTraceDispatcherOptions options_;
};

IRTraceDispatcher::ptr_t createIRTraceDispatcher(exec::static_thread_pool& thread_pool,
                                                 const IRTraceDispatcherOptions& options);

}  // namespace mllm::ir
