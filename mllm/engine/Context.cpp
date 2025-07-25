/**
 * @file Context.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-21
 *
 */
#include "mllm/engine/Context.hpp"
#include "mllm/engine/SessionTCB.hpp"

namespace mllm {

Context::Context() {
  // 1. Add main thread
  main_thread_ = std::make_shared<SessionTCB>();
  main_thread_->system_tid = std::this_thread::get_id();
  session_threads_.insert({main_thread_->system_tid, main_thread_});

  // 2. Add memory manager
  memory_manager_ = std::make_shared<MemoryManager>();

  // 3. Add dispatcher manager
  // TODO
}

void Context::registerBackend(const Backend::ptr_t& new_backend) { backends_.reg(new_backend->device(), new_backend); }

Backend::ptr_t Context::getBackend(const DeviceTypes& device) {
  if (!backends_.has(device)) {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "Backend for device {} not found", deviceTypes2Str(device));
  }
  return backends_[device];
}

uint32_t Context::getUUID() {
  uint32_t ret = custom_uuid_giver_;
  custom_uuid_giver_++;
  return ret;
}

SessionTCB::ptr_t Context::thisThread() {
  if (!session_threads_.count(std::this_thread::get_id())) {
    MLLM_WARN(
        "This control thread did not registered a SessionTCB in Context. The Context will automatically create one for you. "
        "But it is recommend to create SessionTCB manually.");
    session_threads_.insert({std::this_thread::get_id(), std::make_shared<SessionTCB>()});
  }
  return session_threads_[std::this_thread::get_id()];
}

SessionTCB::ptr_t Context::mainThread() { return main_thread_; }

}  // namespace mllm
