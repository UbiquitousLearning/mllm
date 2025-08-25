// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)

namespace mllm::cpu::x86 {

// Should support [B, S, H * D] and [B, S, H, D]
void rmsnorm_fp32(const mllm_fp32_t* __restrict X, const mllm_fp32_t* __restrict W, mllm_fp32_t* __restrict Y, int D,
                  float epsilon, bool add_unit_offset, int thread_count);

}  // namespace mllm::cpu::x86

#endif
