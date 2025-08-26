// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_ARCH_X86) || defined(MLLM_HOST_ARCH_X86_64)

#include <hwy/highway.h>
#include <cmath>
#include <numbers>

namespace mllm::cpu::x86 {

namespace hn = hwy::HWY_NAMESPACE;

// Highway-based math functions for x86
template<class D, class V>
HWY_INLINE V vexpq_fast_f32(D d, V x) {
  // Implementation of fast exponential function using Highway
  const auto c_exp_hi = hn::Set(d, 88.3762626647949f);
  const auto c_exp_lo = hn::Set(d, -88.3762626647949f);
  const auto c_cephes_LOG2EF = hn::Set(d, std::numbers::log2e_v<float>);
  const auto c_cephes_exp_C1 = hn::Set(d, std::numbers::ln2_v<float>);
  const auto c_cephes_exp_C2 = hn::Set(d, -2.12194440e-4f);
  const auto c_cephes_exp_p0 = hn::Set(d, 1.9875691500E-4f);
  const auto c_cephes_exp_p1 = hn::Set(d, 1.3981999507E-3f);
  const auto c_cephes_exp_p2 = hn::Set(d, 8.3334519073E-3f);
  const auto c_cephes_exp_p3 = hn::Set(d, 4.1665795894E-2f);
  const auto c_cephes_exp_p4 = hn::Set(d, 1.6666665459E-1f);
  const auto c_cephes_exp_p5 = hn::Set(d, 5.0000001201E-1f);
  const auto one = hn::Set(d, 1.0f);
  const auto half = hn::Set(d, 0.5f);

  // Clamp x to [c_exp_lo, c_exp_hi]
  x = hn::Min(hn::Max(x, c_exp_lo), c_exp_hi);

  // Express exp(x) as exp(g + n*log(2))
  auto fx = hn::MulAdd(x, c_cephes_LOG2EF, half);

  // Floor
  auto tmp = hn::Floor(fx);

  // If greater, subtract 1
  auto mask = hn::Gt(tmp, fx);
  // Convert mask to vector type before arithmetic operations
  auto mask_vec = hn::IfThenElse(mask, one, hn::Zero(d));
  fx = hn::Sub(tmp, mask_vec);

  auto tmp2 = hn::Mul(fx, c_cephes_exp_C1);
  auto z = hn::Mul(fx, c_cephes_exp_C2);
  x = hn::Sub(hn::Sub(x, tmp2), z);

  z = hn::Mul(x, x);

  auto y = c_cephes_exp_p0;
  y = hn::MulAdd(y, x, c_cephes_exp_p1);
  y = hn::MulAdd(y, x, c_cephes_exp_p2);
  y = hn::MulAdd(y, x, c_cephes_exp_p3);
  y = hn::MulAdd(y, x, c_cephes_exp_p4);
  y = hn::MulAdd(y, x, c_cephes_exp_p5);
  y = hn::MulAdd(y, z, x);
  y = hn::Add(y, one);

  // Build 2^n
  // Use Rebind on the SIMD tag type (D), not the vector type (V)
  using DI = hn::Rebind<int32_t, D>;
  DI di;
  auto int_vec = hn::Add(hn::ConvertTo(di, fx), hn::Set(di, 0x7f));
  auto pow2n = hn::BitCast(d, hn::ShiftLeft<23>(int_vec));

  return hn::Mul(y, pow2n);
}

static inline float vsquare_mean_fp32(const float* __restrict X, int dim) {
  const hn::ScalableTag<float> d;
  auto sum_vec = hn::Zero(d);
  auto square_vec = hn::Zero(d);

  int i = 0;
  for (; i + hn::Lanes(d) <= dim; i += hn::Lanes(d)) {
    auto vec = hn::Load(d, X + i);
    auto squared = hn::Mul(vec, vec);
    sum_vec = hn::Add(sum_vec, squared);
  }

  float acc = hn::ReduceSum(d, sum_vec);
  for (; i < dim; ++i) { acc += X[i] * X[i]; }
  return acc / static_cast<float>(dim);
}

}  // namespace mllm::cpu::x86
#endif
