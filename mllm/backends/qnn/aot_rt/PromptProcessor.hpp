// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/qnn/aot_rt/QnnAOTModule.hpp"
#include "mllm/backends/qnn/aot_rt/KVCacheManager.hpp"
#include "mllm/backends/qnn/aot_rt/QnnAOTConfig.hpp"
#include "mllm/core/Tensor.hpp"
#include <vector>
#include <memory>

namespace mllm::qnn::aot {

template<typename T>
class PromptProcessor {
 public:
  PromptProcessor(KVCacheManager<T>* kv_manager, QnnAOTConfig config);

  /**
   * Prefill an LLM Module with the given text input.
   * @param prompt_tokens The text prompt tokens to the LLM Module.
   * @param start_pos The starting position in KV cache.
   * @return The next token (or logits).
   */
  int64_t prefill(const std::vector<int64_t>& prompt_tokens, int64_t start_pos = 0);

  void init_io();
  void prepare_io(const std::vector<int64_t>& prompt_tokens, int64_t prompt_pos, int64_t start_pos);

 private:
  std::unique_ptr<QnnAOTModule> module_;
  KVCacheManager<T>* kv_manager_;
  QnnAOTConfig config_;

  std::vector<mllm::Tensor> input_tensors_;
  std::vector<mllm::Tensor> output_tensors_;
};

}  // namespace mllm::qnn::aot
