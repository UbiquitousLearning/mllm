// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "mllm/engine/Task.hpp"
#include "mllm/core/DeviceTypes.hpp"

// C++26's Feature. Added by involving NVIDIA's stdexec
#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>

namespace mllm {

class Dispatcher {
 public:
  using ptr_t = std::shared_ptr<Dispatcher>;
  using dispatcher_id_t = int32_t;
  using preprocess_task_func_t = std::function<void(const Task::ptr_t& task)>;
  using afterprocess_task_func_t = std::function<void(const Task::ptr_t& task)>;

  static constexpr int32_t cpu_dispatcher_id = static_cast<int32_t>(DeviceTypes::kCPU);
  static constexpr int32_t cuda_dispatcher_id = static_cast<int32_t>(DeviceTypes::kCUDA);
  static constexpr int32_t opencl_dispatcher_id = static_cast<int32_t>(DeviceTypes::kOpenCL);
  static constexpr int32_t qnn_dispatcher_id = static_cast<int32_t>(DeviceTypes::kQNN);
  static constexpr int32_t trace_dispatcher_id = static_cast<int32_t>(DeviceTypes::kDeviceTypes_End) + 1;
  static constexpr int32_t custom_dispatcher_start_id = static_cast<int32_t>(DeviceTypes::kDeviceTypes_End) + 2;

  explicit Dispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id);

  virtual ~Dispatcher() = default;

  void setPreprocessTaskFunc(const preprocess_task_func_t& func);

  void setAfterprocessTaskFunc(const afterprocess_task_func_t& func);

  virtual void preprocessTask(const Task::ptr_t& task);

  virtual void afterprocessTask(const Task::ptr_t& task);

  virtual void receive(const Task::ptr_t& task);

  // This method should be implemented by launching a new thread to handle the task using thread_pool_
  // e.g. `stdexec::schedule(scheduler) | stdexec::then([this, task] { process(task); });`
  virtual TaskResult::sender_t asyncReceive(const Task::ptr_t& task) = 0;

  virtual void process(const Task::ptr_t& task);

  virtual void syncWait();

  inline dispatcher_id_t id() { return dispatcher_id_; }

 protected:
  dispatcher_id_t dispatcher_id_;

  preprocess_task_func_t preprocess_task_func_ = nullptr;
  afterprocess_task_func_t afterprocess_task_func_ = nullptr;

  exec::static_thread_pool& thread_pool_;
};

}  // namespace mllm
