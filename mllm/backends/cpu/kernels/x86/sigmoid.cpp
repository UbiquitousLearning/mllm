// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/kernels/x86/sigmoid.hpp"
#include "mllm/core/Parallel.hpp"

#if defined(MLLM_HOST_ARCH_X86) || defined(MLLM_HOST_ARCH_X86_64)

#include "mllm/backends/cpu/kernels/common/sigmoid-inl.hpp"
#include <hwy/highway.h>

namespace mllm::cpu::x86 {

namespace hn = hwy::HWY_NAMESPACE;

void sigmoid_fp32(const mllm_fp32_t* __restrict X, mllm_fp32_t* __restrict Y, int len, int thread_count) {
  using D = hn::ScalableTag<float>;
  const D d;
  const auto vector_size = hn::Lanes(d);
  const int aligned_len = len - (len % vector_size);

  if (thread_count > 1) {
    MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(i, 0, aligned_len, vector_size, thread_count) {
      auto x = hn::LoadU(d, X + i);
      auto result = mllm::cpu::common::HWY_NAMESPACE::__sigmoid_fp32_vector(d, x);
      hn::StoreU(result, d, Y + i);
    }
    MLLM_AUTO_PARALLEL_FOR_END_NT()

    // Handle remaining elements
    for (int i = aligned_len; i < len; ++i) { Y[i] = 1.0f / (1.0f + std::exp(-X[i])); }
  } else {
    int i = 0;
    for (; i + vector_size <= len; i += vector_size) {
      auto x = hn::LoadU(d, X + i);
      auto result = mllm::cpu::common::HWY_NAMESPACE::__sigmoid_fp32_vector(d, x);
      hn::StoreU(result, d, Y + i);
    }

    // Handle remaining elements
    for (; i < len; ++i) { Y[i] = 1.0f / (1.0f + std::exp(-X[i])); }
  }
}

}  // namespace mllm::cpu::x86

#endif
