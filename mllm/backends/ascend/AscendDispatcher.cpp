// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/ascend/AscendDispatcher.hpp"
#include "mllm/backends/ascend/AscendBackend.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/engine/Dispatcher.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/tracy_perf/Tracy.hpp"

#ifdef MLLM_PERFETTO_ENABLE
#include "mllm/engine/Perf.hpp"
#endif

namespace mllm::ascend {

AscendDispatcher::AscendDispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id,
                                   const AscendDispatcherOptions& options)
    : Dispatcher(thread_pool, id), options_(options) {}

void AscendDispatcher::receive(const Task::ptr_t& task) {
  switch (task->type) {
    case TaskTypes::kExecuteModule:
    case TaskTypes::kExecuteOp: {
      process(task);
      break;
    }
    default: NYI("Only execute op/module task is supported in AscendDispatcher::receive");
  }
}

TaskResult::sender_t AscendDispatcher::asyncReceive(const Task::ptr_t& task) {
  switch (task->type) {
    case TaskTypes::kExecuteModule: {
      MLLM_EMPTY_SCOPE;
      break;
    }
    default: NYI("Only execute module task is supported in AscendDispatcher::asyncReceive");
  }
  auto scheduler = thread_pool_.get_scheduler();
  return stdexec::schedule(scheduler) | stdexec::then([this, task] { process(task); });
}

void AscendDispatcher::process(const Task::ptr_t& task) {
  MLLM_TRACY_ZONE_SCOPED;
  switch (task->type) {
    case TaskTypes::kExecuteOp: {
      task->op->reshape(task->inputs, task->outputs);
      task->op->setup(task->inputs, task->outputs);
      task->op->forward(task->inputs, task->outputs);
      
      break;
    }
    case TaskTypes::kExecuteModule: {
      auto moduleName = static_cast<nn::Module*>(task->custom_context_ptr)->getModuleName();
#ifdef MLLM_PERFETTO_ENABLE
      MLLM_PERF_TRACE_EVENT("mllm.ascend.execute.", perfetto::DynamicString{moduleName},
                            [&](perfetto::EventContext ctx) {
                              int cnt = 0;
                              for (auto& i : task->inputs) {
                                ctx.AddDebugAnnotation(perfetto::DynamicString{"inputs-" + std::to_string(cnt++)},
                                                       i.shape());
                              }
                            });
#endif
      auto ascendBackend = std::static_pointer_cast<AscendBackend>(Context::instance().getBackend(kAscend));

      task->outputs = ((nn::Module*)(task->custom_context_ptr))->forward(task->inputs, task->args);

      // TODO:
      // ascendBackend->graphExecute(moduleName, task->inputs, task->outputs);
      break;
    }
    default: NYI("AscendDispatcher::process not supported task type");
  }
}

void AscendDispatcher::syncWait() {
  // TODO
}

AscendDispatcher::ptr_t createAscendDispatcher(exec::static_thread_pool& thread_pool,
                                               const AscendDispatcherOptions& options) {
  return std::make_shared<AscendDispatcher>(thread_pool, Dispatcher::ascend_dispatcher_id, options);
}

}  // namespace mllm::ascend


