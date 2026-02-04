// Copyright 2024 mllm Authors
// SPDX-License-Identifier: Apache-2.0
//
// Common header for mllm CPU kernels with Highway SIMD support.

#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>

// TVM FFI for Python interop
#include "tvm/ffi/tvm_ffi.h"
#include "tvm/ffi/container/tensor.h"

namespace mllm_kernel {
namespace cpu {

// Get pointer from Tensor with type checking
template<typename T>
inline T* GetDataPtr(tvm::ffi::Tensor arr) {
  return static_cast<T*>(arr.data_ptr());
}

template<typename T>
inline const T* GetConstDataPtr(const tvm::ffi::Tensor& arr) {
  return static_cast<const T*>(arr.data_ptr());
}

// Get total number of elements in an Tensor
inline size_t GetNumElements(const tvm::ffi::Tensor& arr) { return arr.numel(); }

}  // namespace cpu
}  // namespace mllm_kernel
