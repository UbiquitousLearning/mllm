// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "mllm/engine/Dispatcher.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/engine/io/Types.hpp"

namespace mllm::async::io {

namespace details {
void copy(const Tensor& dst, const Tensor& src);

void promoteMMAPTensor2AnonymousMemoryTensor(const Tensor& dst, const Tensor& src);

void loadAnonymousMemoryTensorFromDisk(Tensor& dst, const std::string& tensor_name, const std::string& file_name,
                                       ModelFileVersion version = ModelFileVersion::kV2);
}  // namespace details

struct CpuMemoryDiskDispatcherOptions {
  MLLM_EMPTY_SCOPE;
};

class CpuMemoryDiskDispatcher final : public Dispatcher {
 public:
  using ptr_t = std::shared_ptr<CpuMemoryDiskDispatcher>;

  explicit CpuMemoryDiskDispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id,
                                   const CpuMemoryDiskDispatcherOptions& options);

  void receive(const Task::ptr_t& task) override;

  TaskResult::sender_t asyncReceive(const Task::ptr_t& task) override;

  void process(const Task::ptr_t& task) override;

  void syncWait() override;

 private:
  CpuMemoryDiskDispatcherOptions options_;
};

CpuMemoryDiskDispatcher::ptr_t createCpuMemoryDiskDispatcher(exec::static_thread_pool& thread_pool,
                                                             const CpuMemoryDiskDispatcherOptions& options);

}  // namespace mllm::async::io
