// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <chrono>
#include <fstream>
#include <sstream>

#include "mllm/engine/Context.hpp"
#include "mllm/engine/SessionTCB.hpp"
#include "mllm/engine/DispatcherManager.hpp"

namespace mllm {

PerfGuard::PerfGuard() { Context::instance().setPerfMode(true); }

PerfGuard::~PerfGuard() { Context::instance().setPerfMode(false); }

PerfFile::PerfFile() { perf_json_["init_time"] = Context::instance().curTime(); }

void PerfFile::finalize() {
  std::vector<std::pair<uint32_t, PerfMemoryBlob>> sorted_mem_blobs(mem_blobs_.begin(), mem_blobs_.end());
  std::sort(sorted_mem_blobs.begin(), sorted_mem_blobs.end(),
            [](const auto& a, const auto& b) { return a.second.start_time < b.second.start_time; });

  nlohmann::json memory_json = nlohmann::json::array();
  for (const auto& [uuid, blob] : sorted_mem_blobs) {
    nlohmann::json mem_entry;
    mem_entry["uuid"] = uuid;
    mem_entry["start_time"] = blob.start_time;
    mem_entry["end_time"] = blob.end_time;
    mem_entry["memory_usage"] = blob.memory_usage;
    mem_entry["device_type"] = deviceTypes2Str(blob.device_type);
    memory_json.push_back(mem_entry);
  }
  perf_json_["memory_blobs"] = memory_json;
}

void PerfFile::save(const std::string& filename) {
  std::ofstream file(filename);
  if (file.is_open()) {
    file << perf_json_.dump(4);
    file.close();
  } else {
    MLLM_WARN("Unable to open file {} for writing perf data", filename);
  }
}

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

void Context::setPerfMode(bool perf_mode) {
  perf_mode_ = perf_mode;
  if (perf_mode) {
    MLLM_WARN("Perf mode is enabled. Init a new perf file.");
    perf_file_ = std::make_shared<PerfFile>();
  }
}

bool Context::isPerfMode() { return perf_mode_; }

PerfFile::ptr_t Context::getPerfFile() { return perf_file_; }

uint64_t Context::curTime() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}

std::unordered_map<std::thread::id, SessionTCB::ptr_t> Context::refSessionThreads() { return session_threads_; }

}  // namespace mllm
