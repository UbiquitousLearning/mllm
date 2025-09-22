// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include "mllm/core/Storage.hpp"

namespace mllm {

class Allocator {
 public:
  using ptr_t = std::shared_ptr<Allocator>;

  virtual bool ctrlByMemManager() = 0;

  virtual bool alloc(Storage* storage) = 0;

  virtual bool alloc(const Storage::ptr_t& storage) = 0;

  virtual void free(Storage* storage) = 0;

  virtual void free(const Storage::ptr_t& storage) = 0;

  virtual bool generalAlloc(void** ptr, size_t cap, size_t align) = 0;

  virtual void generalFree(void* ptr) = 0;

  virtual size_t allocSize(Storage* storage) = 0;

  virtual size_t allocSize(const Storage::ptr_t& storage) = 0;

  [[nodiscard]] virtual size_t alignSize() const = 0;
};

}  // namespace mllm
