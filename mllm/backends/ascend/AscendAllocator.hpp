// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/base/Allocator.hpp"
#include "mllm/core/Storage.hpp"

#include <unordered_map>
#include <mutex>


namespace mllm::ascend {

class AscendAllocator final : public Allocator {
 public:
  AscendAllocator();
  ~AscendAllocator();

  inline bool ctrlByMemManager() override { return false; }

  bool alloc(Storage* storage) override;

  bool alloc(const Storage::ptr_t& storage) override;

  void free(Storage* storage) override;

  void free(const Storage::ptr_t& storage) override;

  bool generalAlloc(void** ptr, size_t cap, size_t align) override;

  void generalFree(void* ptr) override;

  size_t allocSize(Storage* storage) override;

  size_t allocSize(const Storage::ptr_t& storage) override;

  [[nodiscard]] size_t alignSize() const override;

private:
  std::mutex block_map_mutex_;
  std::unordered_map<void*, int> storage_to_block_id_;  // Storage ptr -> block ID

};

std::shared_ptr<AscendAllocator> createAscendAllocator();

}  // namespace mllm::ascend
