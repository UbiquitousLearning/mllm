/**
 * @file CPUDispatcher.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-26
 *
 */
#include "mllm/backends/cpu/CPUDispatcher.hpp"
#include "mllm/engine/Dispatcher.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::cpu {

CPUDispatcher::CPUDispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id, const CPUDispatcherOptions& options)
    : Dispatcher(thread_pool, id), options_(options) {
  queue_depth_ = options.queue_depth_;
  need_async_exec_ = options.need_async_exec_;
}

void CPUDispatcher::receive(const Task::ptr_t& task) {
  if (options_.queue_depth_) { MLLM_WARN("CPUDispatcher does not support queue depth, default to 0"); }

  // Start execute
  auto scheduler = thread_pool_.get_scheduler();

  // Begin task
  stdexec::sender auto begin = stdexec::schedule(scheduler);
  stdexec::sender auto again = stdexec::then(begin, [this, task] { process(task); });

  stdexec::sync_wait(std::move(again));
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
    default: NYI("CPUDispatcher::process not supported task type");
  }
}

void CPUDispatcher::syncWait() {
  // FIXME: Only works on queue_depth_ != 0 cases.
  if (options_.queue_depth_) { MLLM_WARN("CPUDispatcher does not support queue depth, default to 0"); }
}

CPUDispatcher::ptr_t createCPUDispatcher(exec::static_thread_pool& thread_pool, const CPUDispatcherOptions& options) {
  return std::make_shared<CPUDispatcher>(thread_pool, Dispatcher::cpu_dispatcher_id, options);
}

}  // namespace mllm::cpu
