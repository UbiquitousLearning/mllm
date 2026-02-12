// Copyright 2024 mllm Authors
// SPDX-License-Identifier: Apache-2.0
//
// Add constant kernel using Highway SIMD.

#include "mllm_kernel/common.h"

namespace mllm_kernel::cpu {

/**
 * @brief TVM FFI callable kernel class for compile-time constant.
 *
 * @tparam Constant The compile-time constant to add
 */
template<int Constant>
void add_constant(tvm::ffi::Tensor dst, tvm::ffi::Tensor src) {
  float* dst_ptr = GetDataPtr<float>(dst);
  const float* src_ptr = GetConstDataPtr<float>(src);
  size_t n = GetNumElements(src);
  printf("xwk: %d\n", n);
}

/**
 * @brief TVM FFI callable kernel class for runtime constant.
 */
void add_constant_runtime(tvm::ffi::Tensor dst, tvm::ffi::Tensor src, float constant) {
  float* dst_ptr = GetDataPtr<float>(dst);
  const float* src_ptr = GetConstDataPtr<float>(src);
  size_t n = GetNumElements(src);
}

}  // namespace mllm_kernel::cpu
