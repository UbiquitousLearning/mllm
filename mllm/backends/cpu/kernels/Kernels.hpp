/**
 * @file Kernels.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-26
 *
 */
#pragma once

#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
#include "mllm/backends/cpu/kernels/x86/fill.hpp"  // IWYU pragma: export
#endif

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include "mllm/backends/cpu/kernels/arm/fill.hpp"         // IWYU pragma: export
#include "mllm/backends/cpu/kernels/arm/elementwise.hpp"  // IWYU pragma: export
#include "mllm/backends/cpu/kernels/arm/reduce.hpp"       // IWYU pragma: export
#include "mllm/backends/cpu/kernels/arm/transpose.hpp"    // IWYU pragma: export
#include "mllm/backends/cpu/kernels/arm/permute.hpp"      // IWYU pragma: export
#endif
