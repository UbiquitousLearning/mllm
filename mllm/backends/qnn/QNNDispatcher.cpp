// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/QNNDispatcher.hpp"
#include "mllm/engine/Dispatcher.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/nn/Module.hpp"

#ifdef MLLM_PERFETTO_ENABLE
#include "mllm/engine/Perf.hpp"
#endif

namespace mllm::qnn {

QNNDispatcher::QNNDispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id, const QNNDispatcherOptions& options)
    : Dispatcher(thread_pool, id), options_(options) {}

void QNNDispatcher::receive(const Task::ptr_t& task) {
  switch (task->type) {
    case TaskTypes::kExecuteOp: {
      process(task);
      break;
    }
    default: NYI("Only execute op task is supported receive");
  }
}

TaskResult::sender_t QNNDispatcher::asyncReceive(const Task::ptr_t& task) {
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

void QNNDispatcher::process(const Task::ptr_t& task) {
  switch (task->type) {
    case TaskTypes::kExecuteOp: {
      auto op = task->op;
      auto& inputs = task->inputs;
      auto& outputs = task->outputs;
      op->reshape(inputs, outputs);
      NYI("Qnn Dispatcher::process only handle reshape");
      break;
    }
    case TaskTypes::kExecuteModule: {
#ifdef MLLM_PERFETTO_ENABLE
      auto moduleName = static_cast<nn::Module*>(task->custom_context_ptr)->getModuleName();
      MLLM_PERF_TRACE_EVENT("mllm.kernel", perfetto::DynamicString{moduleName}, [&](perfetto::EventContext ctx) {
        int cnt = 0;
        for (auto& i : task->inputs) {
          ctx.AddDebugAnnotation(perfetto::DynamicString{"inputs-" + std::to_string(cnt++)}, i.shape());
        }
      });
#endif
      task->outputs = ((nn::Module*)(task->custom_context_ptr))->__main(task->inputs, task->args);
      NYI("There should call qnn graph");
      break;
    }
    default: NYI("QNNDispatcher::process not supported task type");
  }
}

void QNNDispatcher::syncWait() {
  // TODO
}

QNNDispatcher::ptr_t createQNNDispatcher(exec::static_thread_pool& thread_pool, const QNNDispatcherOptions& options) {
  return std::make_shared<QNNDispatcher>(thread_pool, Dispatcher::cpu_dispatcher_id, options);
}

}  // namespace mllm::qnn
