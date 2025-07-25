/**
 * @file Task.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-25
 *
 */
#pragma once

#include <cstdint>
#include <memory>

namespace mllm {

enum class TaskTypes : int32_t {

};

class Task {
 public:
  using ptr_t = std::shared_ptr<Task>;

 private:
};

}  // namespace mllm
