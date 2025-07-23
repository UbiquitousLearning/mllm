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

#ifndef __ARM_NEON
#error Mllm's Arm backend only support those devices that have neon support
#endif

#if __ARM_ARCH < 8
#error Mllm's Arm backend only support those devices that have armv8 or above
#endif

namespace mllm::arm {

class ArmBackend final : public Backend {
 public:
  using ptr_t = std::shared_ptr<ArmBackend>;

 private:
};

std::shared_ptr<ArmBackend> createArmBackend();

}  // namespace mllm::arm
