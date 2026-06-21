// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/CPUDispatcher.hpp"
#include "mllm/engine/Dispatcher.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/tracy_perf/Tracy.hpp"

#ifdef MLLM_PERFETTO_ENABLE
#include "mllm/engine/Perf.hpp"
#endif

namespace mllm::cpu {

CPUDispatcher::CPUDispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id, const CPUDispatcherOptions& options)
    : Dispatcher(thread_pool, id), options_(options) {}

void CPUDispatcher::receive(const Task::ptr_t& task) {
  switch (task->type) {
    case TaskTypes::kExecuteModule:
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
  MLLM_TRACY_ZONE_SCOPED;
  switch (task->type) {
    case TaskTypes::kExecuteOp: {
#ifdef MLLM_PERFETTO_ENABLE
      auto op_name = optype2Str(task->op->getOpType());
      MLLM_PERF_TRACE_EVENT("mllm.kernel", perfetto::DynamicString{optype2Str(task->op->getOpType())},
                            [&](perfetto::EventContext ctx) {
                              int cnt = 0;
                              for (auto& i : task->inputs) {
                                ctx.AddDebugAnnotation(perfetto::DynamicString{"inputs-" + std::to_string(cnt++)}, i.shape());
                              }
                            });
#endif
      auto op = task->op;
      auto& inputs = task->inputs;
      auto& outputs = task->outputs;
      op->reshape(inputs, outputs);
      op->setup(inputs, outputs);
      op->forward(inputs, outputs);
      break;
    }
    case TaskTypes::kExecuteModule: {
      task->outputs = ((nn::Module*)(task->custom_context_ptr))->forward(task->inputs, task->args);
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
