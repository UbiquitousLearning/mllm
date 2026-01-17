// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/aot_rt/QnnAOTModule.hpp"
#include "mllm/backends/qnn/aot_rt/KVCacheManager.hpp"
#include "mllm/backends/qnn/aot_rt/PromptProcessor.hpp"
#include "mllm/backends/qnn/aot_rt/TokenGenerator.hpp"
#include "mllm/preprocessor/tokenizers/AutoTokenizer.hpp"
#include <string>
#include <memory>
#include <vector>
#include <functional>

namespace mllm::qnn::aot {

struct RunnerConfig {
  std::string model_path;
  float temperature = 0.8f;
  int num_layers = 28;
  int num_heads = 12;
  int head_dim = 128;
  int vocab_size = 151936;
  int context_len = 4096;
  int ar_len = 128;  // Chunk size for prefill
};

class Runner {
 public:
  explicit Runner(const RunnerConfig& config, mllm::preprocessor::AutoTokenizer* tokenizer);
  ~Runner() = default;

  bool load();
  void generate(std::vector<uint64_t>& prompt_tokens, int32_t seq_len,
                const std::function<void(const std::string&)>& token_callback);

 private:
  RunnerConfig config_;
  mllm::preprocessor::AutoTokenizer* tokenizer_;

  std::unique_ptr<KVCacheManager<uint8_t>> kv_manager_;
  std::unique_ptr<PromptProcessor<uint8_t>> prompt_processor_;
  std::unique_ptr<TokenGenerator<uint8_t>> token_generator_;
};

}  // namespace mllm::qnn::aot