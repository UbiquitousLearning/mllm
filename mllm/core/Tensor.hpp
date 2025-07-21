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
 private:
  std::shared_ptr<TensorViewImpl> impl_ = nullptr;
  std::unordered_map<std::string, TensorViewImpl> attached_views_;
};

}  // namespace mllm
