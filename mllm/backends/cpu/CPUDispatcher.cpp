// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/CPUDispatcher.hpp"
#include "mllm/engine/Dispatcher.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/nn/Module.hpp"

namespace mllm::cpu {

CPUDispatcher::CPUDispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id, const CPUDispatcherOptions& options)
    : Dispatcher(thread_pool, id), options_(options) {}

void CPUDispatcher::receive(const Task::ptr_t& task) {
  switch (task->type) {
    case TaskTypes::kExecuteOp: {
      process(task);
      break;
    }
    default: NYI("Only execute op task is supported receive");
  }
}

TaskResult::sender_t CPUDispatcher::asyncReceive(const Task::ptr_t& task) {
  switch (task->type) {
    case TaskTypes::kExecuteModule: {
      MLLM_EMPTY_SCOPE;
      break;
    }
    default: NYI("Only execute module task is supported asyncReceive");
  }
  auto scheduler = thread_pool_.get_scheduler();
  return stdexec::schedule(scheduler) | stdexec::then([this, task] { process(task); });
}

void CPUDispatcher::process(const Task::ptr_t& task) {
  switch (task->type) {
    case TaskTypes::kExecuteOp: {
      auto op = task->op;
      auto& inputs = task->inputs;
      auto& outputs = task->outputs;
      op->reshape(inputs, outputs);
      op->setup(inputs, outputs);
      op->forward(inputs, outputs);
      break;
    }
    case TaskTypes::kExecuteModule: {
      task->outputs = ((nn::Module*)(task->custom_context_ptr))->__main(task->inputs);
      break;
    }
    default: NYI("CPUDispatcher::process not supported task type");
  }
}

void CPUDispatcher::syncWait() {
  // TODO
}

CPUDispatcher::ptr_t createCPUDispatcher(exec::static_thread_pool& thread_pool, const CPUDispatcherOptions& options) {
  return std::make_shared<CPUDispatcher>(thread_pool, Dispatcher::cpu_dispatcher_id, options);
}

}  // namespace mllm::cpu
