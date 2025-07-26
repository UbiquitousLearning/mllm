/**
 * @file Task.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-25
 *
 */
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
