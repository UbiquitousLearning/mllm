/**
 * @file ArmBackend.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-22
 *
 */
#pragma once

#include "mllm/backends/base/Backend.hpp"

namespace mllm::arm {

class ArmBackend final : public Backend {
 public:
  using ptr_t = std::shared_ptr<ArmBackend>;

 private:
};

}  // namespace mllm::arm
