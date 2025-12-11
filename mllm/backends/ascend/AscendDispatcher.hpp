// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "mllm/engine/Dispatcher.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::ascend {

struct AscendDispatcherOptions {
  MLLM_EMPTY_SCOPE;
};

class AscendDispatcher final : public Dispatcher {
 public:
  using ptr_t = std::shared_ptr<AscendDispatcher>;

  explicit AscendDispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id,
                            const AscendDispatcherOptions& options);

  void receive(const Task::ptr_t& task) override;

  TaskResult::sender_t asyncReceive(const Task::ptr_t& task) override;

  void process(const Task::ptr_t& task) override;

  void syncWait() override;

 private:
  AscendDispatcherOptions options_;
};

AscendDispatcher::ptr_t createAscendDispatcher(exec::static_thread_pool& thread_pool,
                                               const AscendDispatcherOptions& options);

}  // namespace mllm::ascend


