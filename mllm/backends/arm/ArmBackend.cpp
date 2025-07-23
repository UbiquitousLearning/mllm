/**
 * @file ArmBackend.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#include "mllm/backends/arm/ArmBackend.hpp"

namespace mllm::arm {

std::shared_ptr<ArmBackend> createArmBackend() { return std::make_shared<ArmBackend>(); }

}  // namespace mllm::arm
