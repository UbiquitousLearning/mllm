/**
 * @file macro.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-28
 *
 */
#pragma once

namespace mllm::cpu::arm {

#define MLLM_CPU_ARM_FORCE_INLINE __attribute__((always_inline)) inline

}  // namespace mllm::cpu::arm
