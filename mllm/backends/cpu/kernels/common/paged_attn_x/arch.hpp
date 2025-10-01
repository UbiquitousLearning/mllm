// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cassert>
#include "mllm/utils/Common.hpp"

namespace mllm::cpu::paged_attn_x::details {

struct __AnyArchTag {};
using any_arch_tag = __AnyArchTag;

struct __X86ArchTag {};
using x86_arch_tag = __X86ArchTag;

struct __ArmArchTag {};
using arm_arch_tag = __ArmArchTag;

template<typename __ArgTag, typename __LhsDataType, typename __RhsDataType, typename __DstDataType>
struct VectorDotProduct {
  static MLLM_FORCE_INLINE void run(const __LhsDataType* __restrict__ __lhs, const __RhsDataType* __restrict__ __rhs,
                                    __DstDataType* __out, size_t len) {}
};

}  // namespace mllm::cpu::paged_attn_x::details
