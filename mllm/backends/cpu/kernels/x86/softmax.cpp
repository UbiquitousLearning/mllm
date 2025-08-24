// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/x86/softmax.hpp"

#if defined(MLLM_HOST_ARCH_X86) || defined(MLLM_HOST_ARCH_X86_64)

#include <hwy/highway.h>
#include <cmath>
#include <algorithm>
#include <limits>

#include "mllm/backends/cpu/kernels/x86/math.hpp"

namespace mllm::cpu::x86 {
namespace hn = hwy::HWY_NAMESPACE;

void softmax_v1_fp32(const mllm_fp32_t* __restrict X, mllm_fp32_t* __restrict Y, int len, int stride, int thread_count) {
  if (stride != 1 || len <= 16) {
    float max_value = std::numeric_limits<float>::lowest();
    for (int i = 0; i < len; ++i) { max_value = std::max(max_value, X[i * stride]); }
    float sum = 0.f;
    for (int i = 0; i < len; ++i) {
      auto tmp = expf(X[i * stride] - max_value);
      Y[i * stride] = tmp;
      sum += tmp;
    }
    sum = 1.f / sum;
    for (int i = 0; i < len; ++i) { Y[i * stride] *= sum; }
    return;
  }

  const hn::ScalableTag<float> d;
  using V = hn::Vec<decltype(d)>;
  int i = 0;
  V max_vec = hn::Set(d, std::numeric_limits<float>::lowest());
  for (; i + hn::Lanes(d) <= len; i += hn::Lanes(d)) {
    const V x_vec = hn::Load(d, X + i);
    max_vec = hn::Max(max_vec, x_vec);
  }
  float max_value = hn::ReduceMax(d, max_vec);
  for (; i < len; ++i) { max_value = std::max(max_value, X[i]); }
  V sum_vec = hn::Zero(d);
  const V max_vec_broadcast = hn::Set(d, max_value);
  i = 0;
  for (; i + hn::Lanes(d) <= len; i += hn::Lanes(d)) {
    const V x_vec = hn::Load(d, X + i);
    const V normalized = hn::Sub(x_vec, max_vec_broadcast);
    const V exp_vec = mllm::cpu::x86::vexpq_fast_f32(d, normalized);
    hn::Store(exp_vec, d, Y + i);
    sum_vec = hn::Add(sum_vec, exp_vec);
  }
  float sum_value = hn::ReduceSum(d, sum_vec);
  for (; i < len; ++i) {
    float tmp = expf(X[i] - max_value);
    Y[i] = tmp;
    sum_value += tmp;
  }
  sum_value = 1.f / sum_value;
  const V inv_sum_vec = hn::Set(d, sum_value);
  i = 0;
  for (; i + hn::Lanes(d) <= len; i += hn::Lanes(d)) {
    const V y_vec = hn::Load(d, Y + i);
    const V result = hn::Mul(y_vec, inv_sum_vec);
    hn::Store(result, d, Y + i);
  }
  for (; i < len; ++i) { Y[i] *= sum_value; }
}

}  // namespace mllm::cpu::x86

#endif
