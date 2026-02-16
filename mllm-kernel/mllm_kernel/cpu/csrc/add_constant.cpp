// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Add constant kernel using Highway SIMD.

#include <mllm_kernel/tensor.hpp>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <mllm_kernel/utils.hpp>   // For div_ceil, RuntimeCheck
#include <tvm/ffi/container/tensor.h>

#include <tuple>

// >>>> Highway dynamic dispatch setup.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "../csrc/add_constant.cpp"
#include <hwy/foreach_target.h>  // IWYU pragma: keep
// <<<< Highway dynamic dispatch setup.

#include <mllm_kernel/cpu/add_constant.hpp>

#if HWY_ONCE
#include <hwy/targets.cc>  // Provide GetChosenTarget for JIT-built module.
#endif

namespace mllm_kernel::cpu {

#if HWY_ONCE
HWY_EXPORT(add_constant_ct_1);
HWY_EXPORT(add_constant_ct_2);
HWY_EXPORT(add_constant_ct_4);
HWY_EXPORT(add_constant_ct_8);
HWY_EXPORT(add_constant_ct_16);
HWY_EXPORT(add_constant_rt);

namespace detail {

inline void dispatch_add_constant_ct_1(float* dst, const float* src, std::size_t num_elements) {
  HWY_DYNAMIC_DISPATCH(add_constant_ct_1)(dst, src, num_elements);
}

inline void dispatch_add_constant_ct_2(float* dst, const float* src, std::size_t num_elements) {
  HWY_DYNAMIC_DISPATCH(add_constant_ct_2)(dst, src, num_elements);
}

inline void dispatch_add_constant_ct_4(float* dst, const float* src, std::size_t num_elements) {
  HWY_DYNAMIC_DISPATCH(add_constant_ct_4)(dst, src, num_elements);
}

inline void dispatch_add_constant_ct_8(float* dst, const float* src, std::size_t num_elements) {
  HWY_DYNAMIC_DISPATCH(add_constant_ct_8)(dst, src, num_elements);
}

inline void dispatch_add_constant_ct_16(float* dst, const float* src, std::size_t num_elements) {
  HWY_DYNAMIC_DISPATCH(add_constant_ct_16)(dst, src, num_elements);
}

inline void dispatch_add_constant_runtime(float* dst, const float* src, std::size_t num_elements, float constant) {
  HWY_DYNAMIC_DISPATCH(add_constant_rt)(dst, src, num_elements, constant);
}

inline auto prepare_kernel_args(tvm::ffi::TensorView dst,
                                tvm::ffi::TensorView src) -> std::tuple<float*, const float*, std::size_t> {
  using namespace mllm_kernel::host;  // NOLINT
  SymbolicSize N = {"num_elements"};
  SymbolicDevice device_;
  (void)TensorMatcher({N})           // 1D tensor, must be contiguous
      .with_dtype<float>()           // must be float32
      .with_device<kDLCPU>(device_)  // must be on CPU device
      .verify(dst)                   // check tensor dst
      .verify(src);                  // check tensor src

  const auto num_elements_i64 = N.unwrap();
  RuntimeCheck(num_elements_i64 >= 0, "num_elements must be non-negative, got ", num_elements_i64);
  auto* dst_ptr = static_cast<float*>(dst.data_ptr());
  const auto* src_ptr = static_cast<const float*>(src.data_ptr());
  return {dst_ptr, src_ptr, static_cast<std::size_t>(num_elements_i64)};
}

}  // namespace detail

template<int Constant>
void add_constant(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
  static_assert(Constant == 1 || Constant == 2 || Constant == 4 || Constant == 8 || Constant == 16,
                "Only constants {1, 2, 4, 8, 16} are supported");

  const auto [dst_ptr, src_ptr, num_elements] = detail::prepare_kernel_args(dst, src);
  if constexpr (Constant == 1) {
    detail::dispatch_add_constant_ct_1(dst_ptr, src_ptr, num_elements);
  } else if constexpr (Constant == 2) {
    detail::dispatch_add_constant_ct_2(dst_ptr, src_ptr, num_elements);
  } else if constexpr (Constant == 4) {
    detail::dispatch_add_constant_ct_4(dst_ptr, src_ptr, num_elements);
  } else if constexpr (Constant == 8) {
    detail::dispatch_add_constant_ct_8(dst_ptr, src_ptr, num_elements);
  } else {
    detail::dispatch_add_constant_ct_16(dst_ptr, src_ptr, num_elements);
  }
}

void add_constant_runtime(tvm::ffi::TensorView dst, tvm::ffi::TensorView src, float constant) {
  const auto [dst_ptr, src_ptr, num_elements] = detail::prepare_kernel_args(dst, src);
  detail::dispatch_add_constant_runtime(dst_ptr, src_ptr, num_elements, constant);
}

#endif  // HWY_ONCE

}  // namespace mllm_kernel::cpu
