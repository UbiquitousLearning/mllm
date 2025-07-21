/**
 * @file Tensor.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-21
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "mllm/core/TensorViewImpl.hpp"

namespace mllm {

class Tensor {
 public:
  /**
   * @brief  Create a nil tensor
   *
   * @return Tensor
   */
  static inline Tensor nil() { return {}; };

  /**
   * @brief If this tensor is not initialized
   *
   * @note explicit must be set to avoid auto i = tensor. But i is set as bool type.
   *
   * @return true
   * @return false
   */
  explicit inline operator bool() const noexcept { return impl_ != nullptr; }

 private:
  std::shared_ptr<TensorViewImpl> impl_ = nullptr;
  std::unordered_map<std::string, TensorViewImpl> attached_views_;
};

}  // namespace mllm
