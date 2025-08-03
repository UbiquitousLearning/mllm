// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>

#include "mllm/backends/cpu/ops/EmbeddingOp.hpp"
#include "mllm/core/DataTypes.hpp"

namespace mllm::cpu {

CPUEmbeddingOp::CPUEmbeddingOp(const aops::EmbeddingOpOptions& options) : aops::EmbeddingOp(options) {}

void CPUEmbeddingOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ins = inputs[0];
  auto ous = outputs[0];

  auto B = ins.shape()[0];
  auto S = ins.shape()[1];

  auto weight_dtype = weight_.dtype();

  MLLM_RT_ASSERT_EQ(ins.dtype(), kInt64);

  for (int b = 0; b < B; ++b) {
#pragma omp parallel for schedule(auto) num_threads(options_.getThreads()) if (options_.getThreads() > 1)
    for (int s = 0; s < S; ++s) {
      switch (weight_dtype) {
        case kFloat32:
          std::memcpy(ous.offsettedPtr<char>({b, s, 0}),
                      weight_.ptr<mllm_fp32_t>() + options_.hidden_size * (*ins.offsettedPtr<mllm_int64_t>({b, s})),
                      options_.hidden_size * sizeof(float));
          break;
        case kFloat16:
          std::memcpy(ous.offsettedPtr<char>({b, s, 0}),
                      weight_.ptr<mllm_fp16_t>() + options_.hidden_size * (*ins.offsettedPtr<mllm_int64_t>({b, s})),
                      options_.hidden_size * sizeof(mllm_fp16_t));
          break;
        default: NYI("Not supported weight dtype for arm llm embedding token op");
      }
    }
  }
}

}  // namespace mllm::cpu
