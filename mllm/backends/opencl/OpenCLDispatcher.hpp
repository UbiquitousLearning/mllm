// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "mllm/engine/Dispatcher.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::opencl {

struct OpenCLDispatcherOptions {
  MLLM_EMPTY_SCOPE;
};

class OpenCLDispatcher final : public Dispatcher {
 public:
  using ptr_t = std::shared_ptr<OpenCLDispatcher>;

  explicit OpenCLDispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id, const OpenCLDispatcherOptions& options);

  void receive(const Task::ptr_t& task) override;

  TaskResult::sender_t asyncReceive(const Task::ptr_t& task) override;

  void process(const Task::ptr_t& task) override;

  void syncWait() override;

 private:
  OpenCLDispatcherOptions options_;
};

OpenCLDispatcher::ptr_t createOpenCLDispatcher(exec::static_thread_pool& thread_pool, const OpenCLDispatcherOptions& options);

}  // namespace mllm::opencl
