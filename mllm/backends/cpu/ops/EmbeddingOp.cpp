// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include <atomic>

#include "mllm/backends/cpu/ops/EmbeddingOp.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Parallel.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize_q4.hpp"

namespace mllm::cpu {

CPUEmbeddingOp::CPUEmbeddingOp(const aops::EmbeddingOpOptions& options) : aops::EmbeddingOp(options) {}

void CPUEmbeddingOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& ins = inputs[0];
  const auto& ous = outputs[0];

  auto B = ins.shape()[0];
  auto S = ins.shape()[1];

  auto weight_dtype = weight_.dtype();

  MLLM_RT_ASSERT_EQ(ins.dtype(), kInt64);
  MLLM_RT_ASSERT(options_.vocab_size > 0);
  MLLM_RT_ASSERT(options_.hidden_size > 0);

  static std::atomic<bool> warned_token_oob{false};

  const bool use_parallel = options_.getThreads() > 1;
  const int thread_count = options_.getThreads();
  for (int b = 0; b < B; ++b) {
    MLLM_CONDITIONAL_PARALLEL_FOR(use_parallel, thread_count, s, 0, S, 1, {
      const auto token_idx = *ins.coffsettedPtr<mllm_int64_t>({b, (int)s});
      auto* out_ptr = ous.coffsettedPtr<char>({b, (int)s, 0});
      if (token_idx < 0 || token_idx >= options_.vocab_size) {
        std::memset(out_ptr, 0, options_.hidden_size * bytesOfType(ous.dtype()));
        bool expected = false;
        if (warned_token_oob.compare_exchange_strong(expected, true)) {
          MLLM_WARN("Embedding token index out of range (idx={}, vocab={}), output row is zero-filled.",
                    token_idx, options_.vocab_size);
        }
      } else {
        switch (weight_dtype) {
          case kFloat32:
            std::memcpy(out_ptr, weight_.ptr<mllm_fp32_t>() + options_.hidden_size * token_idx,
                        options_.hidden_size * sizeof(float));
            break;
          case kFloat16:
            std::memcpy(out_ptr, weight_.ptr<mllm_fp16_t>() + options_.hidden_size * token_idx,
                        options_.hidden_size * sizeof(mllm_fp16_t));
            break;
          case kGGUF_Q4_K: {
            dequantize_row_q4_K(weight_.ptr<block_q4_K>() + token_idx * options_.hidden_size / QK_K,
                                ous.coffsettedPtr<float>({b, (int)s, 0}), options_.hidden_size);
            break;
          }
        
        case kGGUF_Q4_0: {
          auto token_idx = *ins.coffsettedPtr<mllm_int64_t>({b, (int)s});
          if (token_idx >= 0) {
            dequantize_row_q4_0(weight_.ptr<block_q4_0>() + token_idx * options_.hidden_size / QK4_0,
                                ous.coffsettedPtr<float>({b, (int)s, 0}), options_.hidden_size);
          }
          break;
        }
        default: NYI("Not supported weight dtype for arm llm embedding token op");
      }
      }
    });
  }
}

}  // namespace mllm::cpu
