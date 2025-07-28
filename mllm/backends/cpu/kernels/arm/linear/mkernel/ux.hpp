/**
 * @file ux.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-28
 *
 */
#pragma once

#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)

#include "mllm/backends/cpu/kernels/arm/linear/mkernel/u1.hpp"
#include "mllm/backends/cpu/kernels/arm/linear/mkernel/u2.hpp"
#include "mllm/backends/cpu/kernels/arm/linear/mkernel/u3.hpp"
#include "mllm/backends/cpu/kernels/arm/linear/mkernel/u4.hpp"
#include "mllm/backends/cpu/kernels/arm/linear/mkernel/u5.hpp"
#include "mllm/backends/cpu/kernels/arm/linear/mkernel/u6.hpp"
#include "mllm/backends/cpu/kernels/arm/linear/mkernel/u7.hpp"

namespace mllm::cpu::arm {}

#endif