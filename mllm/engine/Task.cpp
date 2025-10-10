// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/engine/Task.hpp"

namespace mllm {

Task::ptr_t Task::createExecuteOpTask(const BaseOp::ptr_t& op, const std::vector<Tensor>& inputs,
                                      const std::vector<Tensor>& outputs) {
  auto task = std::make_shared<Task>();
  task->type = TaskTypes::kExecuteOp;
  task->op = op;
  task->inputs = inputs;
  task->outputs = outputs;
  return task;
}

Task::ptr_t Task::createExecuteModuleTask(void* module_ptr, const std::vector<Tensor>& inputs,
                                          const std::vector<AnyValue>& args) {
  auto task = std::make_shared<Task>();
  task->type = TaskTypes::kExecuteModule;
  task->custom_context_ptr = module_ptr;
  task->inputs = inputs;
  task->args = args;
  return task;
}

}  // namespace mllm
