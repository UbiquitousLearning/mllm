// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "mllm/backends/base/Backend.hpp"
#include "mllm/engine/HpcThreadPool.hpp"

namespace mllm::cpu {

class CPUBackend final : public Backend {
 public:
  ~CPUBackend();

  CPUBackend();

  HpcThreadPool::ptr_t getThreadPool();

  void initThreadPool(int32_t num_threads);

  int32_t taskIndex();

 private:
  int32_t task_index_ = -1;
  HpcThreadPool::ptr_t thread_pool_ = nullptr;
};

std::shared_ptr<CPUBackend> createCPUBackend();

void idleHpcThreadPool();

void wakeupHpcThreadPool();
}  // namespace mllm::cpu
