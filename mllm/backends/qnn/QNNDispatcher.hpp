// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "mllm/engine/Dispatcher.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::qnn {

struct QNNDispatcherOptions {
  MLLM_EMPTY_SCOPE;
};

class QNNDispatcher final : public Dispatcher {
 public:
  using ptr_t = std::shared_ptr<QNNDispatcher>;

  explicit QNNDispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id, const QNNDispatcherOptions& options);

  void receive(const Task::ptr_t& task) override;

  TaskResult::sender_t asyncReceive(const Task::ptr_t& task) override;

  void process(const Task::ptr_t& task) override;

  void syncWait() override;

 private:
  QNNDispatcherOptions options_;
};

QNNDispatcher::ptr_t createQNNDispatcher(exec::static_thread_pool& thread_pool, const QNNDispatcherOptions& options);

}  // namespace mllm::qnn
