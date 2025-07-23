/**
 * @file Allocator.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-22
 *
 */
#pragma once

#include <memory>
#include "mllm/core/Storage.hpp"

namespace mllm {

class Allocator {
 public:
  using ptr_t = std::shared_ptr<Allocator>;

  virtual bool ctrlByMemManager() = 0;

  virtual bool alloc(Storage* storage) = 0;

  virtual bool alloc(const std::shared_ptr<Storage>& storage) = 0;

  virtual void free(Storage* storage) = 0;

  virtual void free(const std::shared_ptr<Storage>& storage) = 0;

  virtual bool generalAlloc(void** ptr, size_t cap, size_t align) = 0;

  virtual void generalFree(void* ptr) = 0;

  virtual size_t allocSize(Storage* storage) = 0;

  virtual size_t allocSize(const std::shared_ptr<Storage>& storage) = 0;

  [[nodiscard]] virtual size_t alignSize() const = 0;
};

}  // namespace mllm
