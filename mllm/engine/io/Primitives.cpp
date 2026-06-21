// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/engine/io/Primitives.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/engine/Dispatcher.hpp"
#include "mllm/engine/io/Types.hpp"

namespace mllm::async::io {
TaskResult::sender_t copy(const Tensor& dst, const Tensor& src) {
  auto task = std::make_shared<Task>();

  task->type = (TaskTypes)AsyncIOTaskTypes::kCopy;
  task->inputs = {src};
  task->outputs = {dst};
  task->args = {};
  task->custom_context_ptr = nullptr;

  auto await_result = Context::instance().dispatcherManager()->asyncSubmit(Dispatcher::cpu_memory_disk_io_dispatcher_id, task);
  return await_result;
}

TaskResult::sender_t promoteMMAPTensor2AnonymousMemoryTensor(const Tensor& dst, const Tensor& src) {
  auto task = std::make_shared<Task>();

  task->type = (TaskTypes)AsyncIOTaskTypes::kPromoteMMAPTensor2AnonymousMemoryTensor;
  task->inputs = {src};
  task->outputs = {dst};
  task->args = {};
  task->custom_context_ptr = nullptr;

  auto await_result = Context::instance().dispatcherManager()->asyncSubmit(Dispatcher::cpu_memory_disk_io_dispatcher_id, task);
  return await_result;
}

TaskResult::sender_t loadAnonymousMemoryTensorFromDisk(Tensor& dst, const std::string& tensor_name,
                                                       const std::string& file_name, ModelFileVersion version) {
  auto task = std::make_shared<Task>();

  task->type = (TaskTypes)AsyncIOTaskTypes::kPromoteMMAPTensor2AnonymousMemoryTensor;
  task->inputs = {dst};
  task->outputs = {dst};
  task->args = {
      AnyValue(std::string(tensor_name)),
      AnyValue(std::string(file_name)),
      AnyValue(version),
  };
  task->custom_context_ptr = nullptr;

  auto await_result = Context::instance().dispatcherManager()->asyncSubmit(Dispatcher::cpu_memory_disk_io_dispatcher_id, task);
  return await_result;
}
}  // namespace mllm::async::io
