// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>

#include "mllm/backends/cpu/ops/EmbeddingOp.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Parallel.hpp"

namespace mllm::cpu {

CPUEmbeddingOp::CPUEmbeddingOp(const aops::EmbeddingOpOptions& options) : aops::EmbeddingOp(options) {}

void CPUEmbeddingOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& ins = inputs[0];
  const auto& ous = outputs[0];

  auto B = ins.shape()[0];
  auto S = ins.shape()[1];

  auto weight_dtype = weight_.dtype();

  MLLM_RT_ASSERT_EQ(ins.dtype(), kInt64);

  const bool use_parallel = options_.getThreads() > 1;
  const int thread_count = options_.getThreads();
  for (int b = 0; b < B; ++b) {
    MLLM_CONDITIONAL_PARALLEL_FOR(use_parallel, thread_count, s, 0, S, 1, {
      switch (weight_dtype) {
        case kFloat32:
          std::memcpy(ous.coffsettedPtr<char>({b, (int)s, 0}),
                      weight_.ptr<mllm_fp32_t>() + options_.hidden_size * (*ins.coffsettedPtr<mllm_int64_t>({b, (int)s})),
                      options_.hidden_size * sizeof(float));
          break;
        case kFloat16:
          std::memcpy(ous.coffsettedPtr<char>({b, (int)s, 0}),
                      weight_.ptr<mllm_fp16_t>() + options_.hidden_size * (*ins.coffsettedPtr<mllm_int64_t>({b, (int)s})),
                      options_.hidden_size * sizeof(mllm_fp16_t));
          break;
        default: NYI("Not supported weight dtype for arm llm embedding token op");
      }
    });
  }
}

}  // namespace mllm::cpu
