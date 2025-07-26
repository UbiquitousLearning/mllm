/**
 * @file X86Dispatcher.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-26
 *
 */
#include "mllm/backends/x86/X86Dispatcher.hpp"
#include "mllm/engine/Dispatcher.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::x86 {

X86Dispatcher::X86Dispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id, const X86DispatcherOptions& options)
    : Dispatcher(thread_pool, id), options_(options) {
  queue_depth_ = options.queue_depth_;
  need_async_exec_ = options.need_async_exec_;
}

void X86Dispatcher::receive(const Task::ptr_t& task) {
  if (options_.queue_depth_) { MLLM_WARN("X86Dispatcher does not support queue depth, default to 0"); }

  // Start execute
  auto scheduler = thread_pool_.get_scheduler();

  // Begin task
  stdexec::sender auto begin = stdexec::schedule(scheduler);
  stdexec::sender auto again = stdexec::then(begin, [this, task] { process(task); });

  stdexec::sync_wait(std::move(again));
}

void X86Dispatcher::process(const Task::ptr_t& task) {
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
    default: NYI("X86Dispatcher::process not supported task type");
  }
}

void X86Dispatcher::syncWait() {
  // FIXME: Only works on queue_depth_ != 0 cases.
  if (options_.queue_depth_) { MLLM_WARN("X86Dispatcher does not support queue depth, default to 0"); }
}

X86Dispatcher::ptr_t createX86Dispatcher(exec::static_thread_pool& thread_pool, const X86DispatcherOptions& options) {
  return std::make_shared<X86Dispatcher>(thread_pool, Dispatcher::cpu_dispatcher_id, options);
}

}  // namespace mllm::x86
