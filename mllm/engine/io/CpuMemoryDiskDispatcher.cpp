// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>

#include "mllm/engine/io/CpuMemoryDiskDispatcher.hpp"
#include "mllm/engine/io/Types.hpp"

namespace mllm::async::io {

namespace details {
void copy(const Tensor& dst, const Tensor& src) {
  MLLM_RT_ASSERT_EQ(dst.device(), DeviceTypes::kCPU);
  MLLM_RT_ASSERT_EQ(src.device(), DeviceTypes::kCPU);
  MLLM_RT_ASSERT_EQ(dst.dtype(), src.dtype());
  MLLM_RT_ASSERT_EQ(dst.shape(), src.shape());
  MLLM_RT_ASSERT(dst.memType() > TensorMemTypes::kParams_End || dst.memType() < TensorMemTypes::kParams_Start);

  // Do not use high-level copy function here. Because we may trap into deadlock.
  auto dst_ptr = dst.ptr<char>();
  auto src_ptr = src.ptr<char>();

  // FIXME: We may need to bind this copy thread on some specific CPU core to get peak performance.

  // Copy
  std::memcpy(dst_ptr, src_ptr, src.bytes());
}

void promoteMMAPTensor2AnonymousMemoryTensor(const Tensor& dst, const Tensor& src) {
  ::mllm::async::io::details::copy(dst, src);
}

void loadAnonymousMemoryTensorFromDisk(Tensor& dst, const std::string& tensor_name, const std::string& file_name,
                                       ModelFileVersion version) {
  ParameterFile::ptr_t p_loader = nullptr;
  if (version == ModelFileVersion::kV1) {
    p_loader = ParameterFileIOImpl<kCPU, ModelFileVersion::kV1>::read(file_name);
  } else if (version == ModelFileVersion::kV2) {
    p_loader = ParameterFileIOImpl<kCPU, ModelFileVersion::kV2>::read(file_name);
  }
  ::mllm::async::io::details::promoteMMAPTensor2AnonymousMemoryTensor(dst, p_loader->pull(tensor_name));
}
}  // namespace details

CpuMemoryDiskDispatcher::CpuMemoryDiskDispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id,
                                                 const CpuMemoryDiskDispatcherOptions& options)
    : Dispatcher(thread_pool, id), options_(options) {}

void CpuMemoryDiskDispatcher::receive(const Task::ptr_t& task) { process(task); }

TaskResult::sender_t CpuMemoryDiskDispatcher::asyncReceive(const Task::ptr_t& task) {
  auto scheduler = thread_pool_.get_scheduler();
  return stdexec::schedule(scheduler) | stdexec::then([this, task] { process(task); });
}

void CpuMemoryDiskDispatcher::process(const Task::ptr_t& task) {
  auto this_task_type = (AsyncIOTaskTypes)task->type;
  switch (this_task_type) {
    case AsyncIOTaskTypes::kCopy: {
      details::copy(task->outputs[0], task->inputs[0]);
      break;
    }
    case AsyncIOTaskTypes::kPromoteMMAPTensor2AnonymousMemoryTensor: {
      details::promoteMMAPTensor2AnonymousMemoryTensor(task->outputs[0], task->inputs[0]);
      break;
    }
    case AsyncIOTaskTypes::kLoadAnonymousMemoryTensorFromDisk: {
      details::loadAnonymousMemoryTensorFromDisk(task->outputs[0], task->args[0].get<std::string>(),
                                                 task->args[1].get<std::string>(), task->args[2].get<ModelFileVersion>());
      break;
    }
  }
}

void CpuMemoryDiskDispatcher::syncWait() {
  // TODO
}

CpuMemoryDiskDispatcher::ptr_t createCpuMemoryDiskDispatcher(exec::static_thread_pool& thread_pool,
                                                             const CpuMemoryDiskDispatcherOptions& options) {
  return std::make_shared<CpuMemoryDiskDispatcher>(thread_pool, Dispatcher::cpu_memory_disk_io_dispatcher_id, options);
}

}  // namespace mllm::async::io
