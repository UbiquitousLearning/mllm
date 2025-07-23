/**
 * @file Backend.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-22
 *
 */
#pragma once

#include <memory>
#include "mllm/backends/base/Allocator.hpp"

namespace mllm {

class Backend {
 public:
  using ptr_t = std::shared_ptr<Backend>;

 private:
  Allocator::ptr_t allocator_ = nullptr;
};

}  // namespace mllm
