/**
 * @file Task.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-25
 *
 */
#pragma once

#include <cstdint>
#include <memory>

#include "mllm/core/BaseOp.hpp"

namespace mllm {

enum class TaskTypes : int32_t {
  kExecuteOp = 0,
};

struct Task {
  using ptr_t = std::shared_ptr<Task>;
  TaskTypes type;
  BaseOp::ptr_t op;
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  void* custom_context_ptr = nullptr;

  static Task::ptr_t createExecuteOpTask(const BaseOp::ptr_t& op, const std::vector<Tensor>& inputs,
                                         const std::vector<Tensor>& outputs);
};

}  // namespace mllm
