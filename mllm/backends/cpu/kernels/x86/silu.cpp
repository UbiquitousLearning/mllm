// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/x86/silu.hpp"
#include "mllm/core/Parallel.hpp"

#if defined(MLLM_HOST_ARCH_X86) || defined(MLLM_HOST_ARCH_X86_64)

#include "mllm/backends/cpu/kernels/x86/math.hpp"
#include <hwy/highway.h>

namespace mllm::cpu::x86 {

namespace hn = hwy::HWY_NAMESPACE;

void silu_fp32(const mllm_fp32_t* __restrict X, mllm_fp32_t* __restrict Y, int len, int thread_count) {
  using V = hn::Vec<hn::ScalableTag<float>>;
  const hn::ScalableTag<float> d;
  const auto zero = hn::Zero(d);
  const auto one = hn::Set(d, 1.0f);

  auto silu_func = [&](auto x) {
    auto neg_x = hn::Neg(x);
    auto exp_neg_x = vexpq_fast_f32(d, neg_x);
    auto sigmoid = hn::Div(one, hn::Add(one, exp_neg_x));
    return hn::Mul(x, sigmoid);
  };

  if (thread_count > 1) {
    const int vector_size = hn::Lanes(d);
    const int aligned_len = len - (len % vector_size);

    MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(i, 0, aligned_len, vector_size, thread_count) {
      V x = hn::LoadU(d, X + i);
      V result = silu_func(x);
      hn::StoreU(result, d, Y + i);
    }
    MLLM_AUTO_PARALLEL_FOR_END_NT()

    // Handle remaining elements
    for (int i = aligned_len; i < len; ++i) { Y[i] = X[i] / (1.0f + std::exp(-X[i])); }
  } else {
    int i = 0;
    const int vector_size = hn::Lanes(d);
    for (; i + vector_size <= len; i += vector_size) {
      V x = hn::LoadU(d, X + i);
      V result = silu_func(x);
      hn::StoreU(result, d, Y + i);
    }

    // Handle remaining elements
    for (; i < len; ++i) { Y[i] = X[i] / (1.0f + std::exp(-X[i])); }
  }
}

}  // namespace mllm::cpu::x86

#endif
