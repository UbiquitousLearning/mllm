// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// NOTE: Highway Kernels for X86 devices is modified from gemma.cpp, we paste the licence of gemma.cpp below:

// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <hwy/highway.h>
#include <hwy/targets.h>
#include <hwy/contrib/math/math-inl.h>
#include <hwy/contrib/algo/transform-inl.h>

HWY_BEFORE_NAMESPACE();
namespace mllm::cpu::common {  // NOLINT
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;
// We use the tanh approximation for gelu (also used in training).
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
//         = 0.5 * x * (1 + tanh(x * (sqrt(2/π) + sqrt(2/π) * 0.044715 * x^2)))
//         = 0.5 * x * (1 + tanh(x * (0.79788 + 0.035677 * x^2)))
//         = x * (0.5 + 0.5 * tanh(x * (0.79788 + 0.035677 * x^2))))
template<class D, HWY_IF_F32_D(D)>
HWY_INLINE hn::Vec<D> __gelu_fp32_vector(D d, hn::Vec<D> v) {
  const hn::Vec<D> kMul = hn::Set(d, 0.03567740813636141f);
  const hn::Vec<D> kSqrt2OverPi = hn::Set(d, 0.797884560804236f);
  const hn::Vec<D> kHalf = hn::Set(d, 0.5f);

  const hn::Vec<D> v2 = hn::Mul(v, v);
  const hn::Vec<D> arg = hn::Mul(v, hn::MulAdd(kMul, v2, kSqrt2OverPi));
  const hn::Vec<D> cdf = hn::MulAdd(kHalf, hn::Tanh(d, arg), kHalf);
  return hn::Mul(v, cdf);
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void gelu_fp32(const float* HWY_RESTRICT x, float* HWY_RESTRICT out, size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  const D d;
  const size_t N = hn::Lanes(d);
  size_t idx = 0;

  // Process full vectors
  for (; idx + N <= size; idx += N) {
    const hn::Vec<D> v = hn::LoadU(d, x + idx);
    const hn::Vec<D> result = __gelu_fp32_vector(d, v);
    hn::StoreU(result, d, out + idx);
  }

  // Process remaining elements
  if (idx < size) {
    const hn::Vec<D> v = hn::LoadN(d, x + idx, size - idx);
    const hn::Vec<D> result = __gelu_fp32_vector(d, v);
    hn::StoreN(result, d, out + idx, size - idx);
  }
}

static HWY_NOINLINE HWY_MAYBE_UNUSED void gelu_fp32_inplace(float* HWY_RESTRICT x, size_t size) {
  namespace hn = hwy::HWY_NAMESPACE;
  using D = hn::ScalableTag<float>;
  hn::Transform(D(), x, size, [](D d, hn::Vec<D> v) HWY_ATTR { return __gelu_fp32_vector(d, v); });
}
}  // namespace HWY_NAMESPACE
}  // namespace mllm::cpu::common
HWY_AFTER_NAMESPACE();
