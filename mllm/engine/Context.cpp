// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <chrono>
#include <sstream>

#include "mllm/engine/Context.hpp"
#include "mllm/engine/SessionTCB.hpp"
#include "mllm/engine/DispatcherManager.hpp"

namespace mllm {

Context& Context::instance() {
  static Context instance;
  return instance;
}

Context::Context() {
  // 1. Add main thread
  main_thread_ = std::make_shared<SessionTCB>();
  main_thread_->system_tid = std::this_thread::get_id();
  session_threads_.insert({main_thread_->system_tid, main_thread_});

  // 2. Add memory manager
  memory_manager_ = std::make_shared<MemoryManager>();

  // 3. Add dispatcher manager
  dispatcher_manager_ = std::make_shared<DispatcherManager>(DispatcherManagerOptions{
      .numa_policy = false,
      .num_threads = 3,
  });
}

void Context::registerBackend(const Backend::ptr_t& new_backend) { backends_.reg(new_backend->device(), new_backend); }

Backend::ptr_t Context::getBackend(const DeviceTypes& device) {
  if (!backends_.has(device)) {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "Backend for device {} not found", deviceTypes2Str(device));
  }
  return backends_[device];
}

std::vector<Tensor> Context::buildOpAndSubmitTask(OpTypes op_type, const BaseOpOptionsBase& base_options,
                                                  const std::vector<Tensor>& inputs, DeviceTypes special_device) {
  auto device = special_device != kDeviceTypes_End ? special_device : inputs[0].device();
  auto op = getBackend(device)->createOp(op_type, base_options);
  auto task = Task::createExecuteOpTask(op, inputs, {});

  auto this_thread = thisThread();
  if (this_thread->trace_mode) {
    // Submit!
    // At this moment, heart pounding like thunder
    // Tasks racing through kernels, swift as lightning
    // Threads await, fate hanging by a thread
    // Success or failure in this one moment
    task->custom_context_ptr = this_thread->ir_context.get();
    dispatcherManager()->submit(Dispatcher::trace_dispatcher_id, task);

    // Everything is Ok. Bravo! You did it.
    // Return what we need.
    return task->outputs;
  } else {
    // Submit!
    // At this moment, heart pounding like thunder
    // Tasks racing through kernels, swift as lightning
    // Threads await, fate hanging by a thread
    // Success or failure in this one moment
    dispatcherManager()->submit(static_cast<int32_t>(device), task);

    // Everything is Ok. Bravo! You did it.
    // Return what we need.
    return task->outputs;
  }
}

uint32_t Context::getUUID() {
  uint32_t ret = custom_uuid_giver_;
  custom_uuid_giver_++;
  return ret;
}

SessionTCB::ptr_t Context::thisThread() {
  if (!session_threads_.count(std::this_thread::get_id())) {
    session_threads_.insert({std::this_thread::get_id(), std::make_shared<SessionTCB>()});
    std::stringstream ss;
    ss << std::this_thread::get_id();
    MLLM_WARN(
        "This control thread did not registered a SessionTCB in Context. The Context will automatically create one for you. "
        "But it is recommend to create SessionTCB manually. THREAD ID: {}",
        ss.str());
  }
  return session_threads_[std::this_thread::get_id()];
}

SessionTCB::ptr_t Context::mainThread() { return main_thread_; }

void Context::setRandomSeed(uint64_t seed) { random_seed_ = seed; }

uint64_t Context::getRandomSeed() { return random_seed_; }

uint64_t Context::curTime() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

std::unordered_map<std::thread::id, SessionTCB::ptr_t> Context::refSessionThreads() { return session_threads_; }

}  // namespace mllm
