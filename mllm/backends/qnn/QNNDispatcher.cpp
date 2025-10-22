// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/QNNDispatcher.hpp"
#include "mllm/backends/qnn/QNNBackend.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/engine/Context.hpp"
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
    case TaskTypes::kExecuteModule: {
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
      // the reshape should be called to init op output tensors
      task->op->reshape(task->inputs, task->outputs);
      // only X2X op is executed in QNN dispatcher
      if (task->op->getOpType() == OpTypes::kX2X || task->op->getOpType() == OpTypes::kEmbedding) {
        task->op->setup(task->inputs, task->outputs);
        task->op->forward(task->inputs, task->outputs);
      }
      break;
    }
    case TaskTypes::kExecuteModule: {
      auto moduleName = static_cast<nn::Module*>(task->custom_context_ptr)->getModuleName();
#ifdef MLLM_PERFETTO_ENABLE
      MLLM_PERF_TRACE_EVENT("mllm.qnn.execute.", perfetto::DynamicString{moduleName}, [&](perfetto::EventContext ctx) {
        int cnt = 0;
        for (auto& i : task->inputs) {
          ctx.AddDebugAnnotation(perfetto::DynamicString{"inputs-" + std::to_string(cnt++)}, i.shape());
        }
      });
#endif
      // here enters in a QNN module, execute it and not dive into its layers
      auto qnnBackend = std::static_pointer_cast<QNNBackend>(Context::instance().getBackend(kQNN));

      task->outputs = ((nn::Module*)(task->custom_context_ptr))->forward(task->inputs, task->args);

      qnnBackend->graphExecute(moduleName, task->inputs, task->outputs);

      break;
    }
    default: NYI("QNNDispatcher::process not supported task type");
  }
}

void QNNDispatcher::syncWait() {
  // TODO
}

QNNDispatcher::ptr_t createQNNDispatcher(exec::static_thread_pool& thread_pool, const QNNDispatcherOptions& options) {
  return std::make_shared<QNNDispatcher>(thread_pool, Dispatcher::qnn_dispatcher_id, options);
}

}  // namespace mllm::qnn
