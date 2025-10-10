// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>

#include "mllm/core/BaseOp.hpp"
#include "mllm/utils/AnyValue.hpp"
#include <exec/any_sender_of.hpp>
#include <stdexec/execution.hpp>

namespace mllm {

template<class... Ts>
using __mllm_any_sender_of = typename exec::any_receiver_ref<stdexec::completion_signatures<Ts...>>::template any_sender<>;

enum class TaskTypes : int32_t {
  kExecuteOp = 0,
  kExecuteModule = 1,
};

struct Task {
  using ptr_t = std::shared_ptr<Task>;

  // Mllm related things
  TaskTypes type;
  BaseOp::ptr_t op;
  std::vector<Tensor> inputs;
  std::vector<Tensor> outputs;
  std::vector<AnyValue> args;
  void* custom_context_ptr = nullptr;

  static Task::ptr_t createExecuteOpTask(const BaseOp::ptr_t& op, const std::vector<Tensor>& inputs,
                                         const std::vector<Tensor>& outputs);
  static Task::ptr_t createExecuteModuleTask(void* module_ptr, const std::vector<Tensor>& inputs,
                                             const std::vector<AnyValue>& args);
};

struct TaskResult {
  using sender_t =
      __mllm_any_sender_of<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr), stdexec::set_stopped_t()>;
};

}  // namespace mllm
