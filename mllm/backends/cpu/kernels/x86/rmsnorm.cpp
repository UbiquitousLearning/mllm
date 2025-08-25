// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/x86/rmsnorm.hpp"

#if defined(MLLM_HOST_ARCH_X86) || defined(MLLM_HOST_ARCH_X86_64)

#include "mllm/backends/cpu/kernels/x86/math.hpp"

#include <hwy/highway.h>
#include <cmath>

namespace mllm::cpu::x86 {
namespace hn = hwy::HWY_NAMESPACE;

void rmsnorm_fp32(const float* __restrict X, const float* __restrict W, float* __restrict Y, int D, float epsilon,
                  bool add_unit_offset, int thread_count) {
  const hn::ScalableTag<float> d;
  auto x_ptr = X;
  auto y_ptr = Y;
  auto w_ptr = W;

  const float rms = 1.0f / std::sqrt(vsquare_mean_fp32(x_ptr, D) + epsilon);
  const auto rms_vec = hn::Set(d, rms);

  if (add_unit_offset) {
    const auto ones = hn::Set(d, 1.0f);
    int i = 0;
    for (; i + hn::Lanes(d) <= D; i += hn::Lanes(d)) {
      auto x_val = hn::Load(d, x_ptr + i);
      auto w_val = hn::Load(d, w_ptr + i);
      auto multiplier = hn::Add(w_val, ones);
      multiplier = hn::Mul(multiplier, rms_vec);
      auto result = hn::Mul(x_val, multiplier);
      hn::Store(result, d, y_ptr + i);
    }
    for (; i < D; ++i) { y_ptr[i] = x_ptr[i] * rms * (w_ptr[i] + 1.0f); }
  } else {
    int i = 0;
    for (; i + hn::Lanes(d) <= D; i += hn::Lanes(d)) {
      auto x_val = hn::Load(d, x_ptr + i);
      auto w_val = hn::Load(d, w_ptr + i);
      auto multiplier = hn::Mul(w_val, rms_vec);
      auto result = hn::Mul(x_val, multiplier);
      hn::Store(result, d, y_ptr + i);
    }
    for (; i < D; ++i) { y_ptr[i] = x_ptr[i] * rms * w_ptr[i]; }
  }
}

}  // namespace mllm::cpu::x86

#endif
