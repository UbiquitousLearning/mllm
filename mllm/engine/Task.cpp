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

}  // namespace mllm
