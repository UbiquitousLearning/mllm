// NOTE: Do NOT use #pragma once here.
// This file is included multiple times by Highway foreach_target dispatch.

#include <hwy/highway.h>

#include <cstddef>

HWY_BEFORE_NAMESPACE();
namespace mllm_kernel::cpu {  // NOLINT
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

inline void add_constant_runtime_simd_impl(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src, std::size_t num_elements,
                                           float constant) {
  const hn::ScalableTag<float> d;
  const std::size_t lanes = hn::Lanes(d);
  const auto v_constant = hn::Set(d, constant);

  std::size_t i = 0;
  for (; i + lanes <= num_elements; i += lanes) {
    const auto v_src = hn::LoadU(d, src + i);
    const auto v_dst = hn::Add(v_src, v_constant);
    hn::StoreU(v_dst, d, dst + i);
  }

  if (i < num_elements) {
    const auto v_src = hn::LoadN(d, src + i, num_elements - i);
    const auto v_dst = hn::Add(v_src, v_constant);
    hn::StoreN(v_dst, d, dst + i, num_elements - i);
  }
}

template<int Constant>
inline void add_constant_simd(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src, std::size_t num_elements) {
  add_constant_runtime_simd_impl(dst, src, num_elements, static_cast<float>(Constant));
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void add_constant_ct_1(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src,
                                                            std::size_t num_elements) {
  add_constant_simd<1>(dst, src, num_elements);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void add_constant_ct_2(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src,
                                                            std::size_t num_elements) {
  add_constant_simd<2>(dst, src, num_elements);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void add_constant_ct_4(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src,
                                                            std::size_t num_elements) {
  add_constant_simd<4>(dst, src, num_elements);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void add_constant_ct_8(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src,
                                                            std::size_t num_elements) {
  add_constant_simd<8>(dst, src, num_elements);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void add_constant_ct_16(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src,
                                                             std::size_t num_elements) {
  add_constant_simd<16>(dst, src, num_elements);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void add_constant_rt(float* HWY_RESTRICT dst, const float* HWY_RESTRICT src,
                                                          std::size_t num_elements, float constant) {
  add_constant_runtime_simd_impl(dst, src, num_elements, constant);
}

}  // namespace HWY_NAMESPACE
}  // namespace mllm_kernel::cpu
HWY_AFTER_NAMESPACE();
