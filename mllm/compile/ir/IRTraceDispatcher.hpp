/**
 * @file IRTraceDispatcher.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-28
 *
 */
#pragma once

#include <memory>

#include "mllm/engine/Dispatcher.hpp"

namespace mllm::ir {

struct IRTraceDispatcherOptions {
  int32_t queue_depth_ = 0;
  bool need_async_exec_ = true;
};

class IRTraceDispatcher final : public Dispatcher {
 public:
  using ptr_t = std::shared_ptr<IRTraceDispatcher>;

  explicit IRTraceDispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id,
                             const IRTraceDispatcherOptions& options);

  void preprocessTask(const Task::ptr_t& task) override;

  void receive(const Task::ptr_t& task) override;

  void process(const Task::ptr_t& task) override;

  void syncWait() override;

 private:
  IRTraceDispatcherOptions options_;
};

IRTraceDispatcher::ptr_t createIRTraceDispatcher(exec::static_thread_pool& thread_pool,
                                                 const IRTraceDispatcherOptions& options);

}  // namespace mllm::ir
