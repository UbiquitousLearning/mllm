// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/aot_rt/QnnAOTRuntime.hpp"
#include <algorithm>
#include <cstring>
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn::aot {
Runner::Runner(const RunnerConfig& config, mllm::preprocessor::AutoTokenizer* tokenizer)
    : config_(config), tokenizer_(tokenizer) {}

bool Runner::load() {
  // init KV cache manager
  KVCacheConfig kv_config;
  kv_config.context_len = config_.context_len;
  kv_config.head_dim = config_.head_dim;

  int32_t prompt_processor_ar_len = config_.ar_len;
  int32_t token_generator_ar_len = 1;

  if (prompt_processor_ar_len == config_.context_len) {
    kv_config.max_cache_len = config_.context_len;
  } else {
    kv_config.max_cache_len = config_.context_len - std::min(token_generator_ar_len, prompt_processor_ar_len);
  }
  kv_config.max_ar_len = std::max(token_generator_ar_len, prompt_processor_ar_len);

  kv_config.num_heads = config_.num_heads;
  kv_config.num_layers = config_.num_layers;

  kv_manager_ = std::make_unique<KVCacheManager<uint8_t>>(kv_config);

  auto backend = mllm::Context::instance().getBackend(mllm::kQNN);
  if (!backend) {
    MLLM_ERROR("QNN Backend not found");
    return false;
  }

  // init prompt processor(prefill)
  PromptProcessor<uint8_t>::Config prefill_config;
  prefill_config.model_path = config_.model_path;
  prefill_config.context_len = config_.context_len;
  prefill_config.num_heads = config_.num_heads;
  prefill_config.num_layers = config_.num_layers;
  prefill_config.ar_len = config_.ar_len;
  prefill_config.vocab_size = config_.vocab_size;
  prefill_config.head_dim = config_.head_dim;
  prefill_config.use_int64_token = false;
  prefill_config.sliding_window = config_.context_len;  // no sliding window for now

  prompt_processor_ = std::make_unique<PromptProcessor<uint8_t>>(kv_manager_.get(), prefill_config);

  // init token generator(decode)
  TokenGenerator<uint8_t>::Config decode_config;
  decode_config.model_path = config_.model_path;
  decode_config.context_len = config_.context_len;
  decode_config.num_heads = config_.num_heads;
  decode_config.num_layers = config_.num_layers;
  decode_config.vocab_size = config_.vocab_size;
  decode_config.head_dim = config_.head_dim;
  decode_config.use_int64_token = false;
  decode_config.sliding_window = config_.context_len;

  // TODO: EOS IDs
  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>();
  eos_ids->insert(151643);
  eos_ids->insert(151645);

  token_generator_ =
      std::make_unique<TokenGenerator<uint8_t>>(tokenizer_, kv_manager_.get(), std::move(eos_ids), decode_config);

  kv_manager_->initCache(backend->allocator().get(), config_.ar_len);
  prompt_processor_->init_io();
  token_generator_->init_io();

  return true;
}

void Runner::generate(std::vector<uint64_t>& prompt_tokens, int32_t seq_len,
                      const std::function<void(const std::string&)>& token_callback) {
  int64_t start_pos = 0;

  std::vector<int64_t> prompt_tokens_i64;
  prompt_tokens_i64.reserve(prompt_tokens.size());
  for (auto t : prompt_tokens) prompt_tokens_i64.push_back((int64_t)t);

  int64_t next_token = prompt_processor_->prefill(prompt_tokens_i64, start_pos);

  prompt_tokens.push_back((uint64_t)next_token);
  if (token_callback) {
    std::wstring wstr = tokenizer_->detokenize(next_token);
    std::string str = mllm::preprocessor::wideString2Utf8String(wstr);
    token_callback(str);
  }

  int64_t cur_pos = prompt_tokens.size();

  token_generator_->generate(prompt_tokens, cur_pos, seq_len, token_callback, false);
}

}  // namespace mllm::qnn::aot
