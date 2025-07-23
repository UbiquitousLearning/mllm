/**
 * @file X86Allocator.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#pragma once

#include "mllm/backends/base/Allocator.hpp"
#include "mllm/core/Storage.hpp"

namespace mllm::x86 {

class X86Allocator final : public Allocator {
 public:
  inline bool ctrlByMemManager() override { return true; }

  bool alloc(Storage* storage) override;

  bool alloc(const Storage::ptr_t& storage) override;

  void free(Storage* storage) override;

  void free(const Storage::ptr_t& storage) override;

  bool generalAlloc(void** ptr, size_t cap, size_t align) override;

  void generalFree(void* ptr) override;

  size_t allocSize(Storage* storage) override;

  size_t allocSize(const Storage::ptr_t& storage) override;

  [[nodiscard]] size_t alignSize() const override;
};

std::shared_ptr<X86Allocator> createX86Allocator();

}  // namespace mllm::x86
