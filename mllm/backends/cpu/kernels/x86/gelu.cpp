// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/CPUArchHelper.hpp"
#include "mllm/core/Parallel.hpp"

#if defined(MLLM_HOST_ARCH_X86) || defined(MLLM_HOST_ARCH_X86_64)

#include <hwy/highway.h>
#include <cmath>
#include "mllm/backends/cpu/kernels/x86/math.hpp"

namespace mllm::cpu::x86 {
namespace hn = hwy::HWY_NAMESPACE;

void gelu_fp32(float* __restrict__ Z, const float* __restrict__ X, int32_t N, int thread_cnt) {
  // TODO
}

void quick_gelu_fp32(float* __restrict__ Z, const float* __restrict__ X, int32_t N, int thread_cnt) {
  // TODO
}

}  // namespace mllm::cpu::x86

#endif
