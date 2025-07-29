/**
 * @file Dispatcher.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-25
 *
 */
#pragma once

#include <memory>

#include "mllm/engine/Task.hpp"
#include "mllm/core/DeviceTypes.hpp"

// C++26's Feature. Added by involving NVIDIA's stdexec
#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>

// Queue
#include <deque>

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

  void setPreprocessTaskFunc(const preprocess_task_func_t& func);

  void setAfterprocessTaskFunc(const afterprocess_task_func_t& func);

  virtual void preprocessTask(const Task::ptr_t& task);

  virtual void afterprocessTask(const Task::ptr_t& task);

  virtual void receive(const Task::ptr_t& task);

  virtual void process(const Task::ptr_t& task);

  virtual void syncWait();

  inline dispatcher_id_t id() { return dispatcher_id_; }

 protected:
  dispatcher_id_t dispatcher_id_;
  std::deque<Task::ptr_t> task_queue_;

  preprocess_task_func_t preprocess_task_func_ = nullptr;
  afterprocess_task_func_t afterprocess_task_func_ = nullptr;

  int32_t queue_depth_ = 0;
  bool need_async_exec_ = true;
  exec::static_thread_pool& thread_pool_;
};

}  // namespace mllm
