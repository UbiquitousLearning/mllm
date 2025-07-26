/**
 * @file X86Dispatcher.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-26
 *
 */
#pragma once

#include <memory>

#include "mllm/engine/Dispatcher.hpp"

namespace mllm::x86 {

struct X86DispatcherOptions {
  int32_t queue_depth_ = 0;
  bool need_async_exec_ = true;
};

class X86Dispatcher final : public Dispatcher {
 public:
  using ptr_t = std::shared_ptr<X86Dispatcher>;

  explicit X86Dispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id, const X86DispatcherOptions& options);

  void receive(const Task::ptr_t& task) override;

  void process(const Task::ptr_t& task) override;

  void syncWait() override;

 private:
  X86DispatcherOptions options_;
};

X86Dispatcher::ptr_t createX86Dispatcher(exec::static_thread_pool& thread_pool, const X86DispatcherOptions& options);

}  // namespace mllm::x86
