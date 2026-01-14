// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/qnn/aot_rt/QnnAOTModule.hpp"
#include "mllm/backends/qnn/aot_rt/KVCacheManager.hpp"
#include "mllm/core/Tensor.hpp"
#include <vector>
#include <memory>
#include <string>

namespace mllm::qnn::aot {

template<typename T>
class PromptProcessor {
 public:
  struct Config {
    std::string model_path;
    int32_t context_len;
    int64_t num_heads;
    int64_t num_layers;
    int32_t ar_len;
    int32_t vocab_size;
    int32_t head_dim;
    bool use_int64_token;
    int sliding_window;
    DataTypes kv_dtype = kUInt8;
  };

  PromptProcessor(KVCacheManager<T>* kv_manager, Config config);

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
  Config config_;

  std::vector<mllm::Tensor> input_tensors_;
  std::vector<mllm::Tensor> output_tensors_;
};

}  // namespace mllm::qnn::aot
