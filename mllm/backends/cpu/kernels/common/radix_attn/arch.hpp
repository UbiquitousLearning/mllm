// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <cassert>
#include "mllm/utils/Common.hpp"

namespace mllm::cpu::radix_attn::details {

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

template<typename __ArgTag, typename __FromDataType, typename __constDataType>
struct MulFromConst {
  static MLLM_FORCE_INLINE void run(__FromDataType* __restrict__ __from, const __constDataType const_v, size_t len) {}
};

template<typename ArchTag, typename T, typename U, typename V>
struct FMAConstArray {
  static MLLM_FORCE_INLINE void run(T* __restrict__ acc_o, const U acc_s, const V* __restrict__ v_token, size_t len) {}
};

template<typename ArchTag, typename T>
struct FilledWithConst {
  static MLLM_FORCE_INLINE void run(T* __restrict__ a, const T v, size_t len) {}
};

}  // namespace mllm::cpu::radix_attn::details
