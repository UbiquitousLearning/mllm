// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/compile/ir/IRTraceDispatcher.hpp"
#include "mllm/engine/Dispatcher.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::ir {

IRTraceDispatcher::IRTraceDispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id,
                                     const IRTraceDispatcherOptions& options)
    : Dispatcher(thread_pool, id), options_(options) {}

void IRTraceDispatcher::preprocessTask(const Task::ptr_t& task) { Dispatcher::preprocessTask(task); }

void IRTraceDispatcher::receive(const Task::ptr_t& task) {
  // Start execute
  auto scheduler = thread_pool_.get_scheduler();

  // Begin task
  stdexec::sender auto begin = stdexec::schedule(scheduler);
  stdexec::sender auto again = stdexec::then(begin, [this, task] { process(task); });

  stdexec::sync_wait(std::move(again));
}

TaskResult::sender_t IRTraceDispatcher::asyncReceive(const Task::ptr_t& task) {
  MLLM_ERROR_EXIT(ExitCode::kCoreError, "asyncReceive is not supported");
  auto scheduler = thread_pool_.get_scheduler();
  return stdexec::schedule(scheduler) | stdexec::then([this, task] { process(task); });
}

void IRTraceDispatcher::process(const Task::ptr_t& task) {
  switch (task->type) {
    case TaskTypes::kExecuteOp: {
      auto op = task->op;
      auto& inputs = task->inputs;
      auto& outputs = task->outputs;
      op->reshape(inputs, outputs);
      op->trace(task->custom_context_ptr, inputs, outputs);
      break;
    }
    default: NYI("IRTraceDispatcher::process not supported task type");
  }
}

void IRTraceDispatcher::syncWait() {
  // TODO
}

IRTraceDispatcher::ptr_t createIRTraceDispatcher(exec::static_thread_pool& thread_pool,
                                                 const IRTraceDispatcherOptions& options) {
  return std::make_shared<IRTraceDispatcher>(thread_pool, Dispatcher::trace_dispatcher_id, options);
}

}  // namespace mllm::ir