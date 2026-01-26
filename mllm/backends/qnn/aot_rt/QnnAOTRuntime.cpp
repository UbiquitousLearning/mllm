// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/aot_rt/QnnAOTRuntime.hpp"
#include <algorithm>
#include <cstring>
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn::aot {
Runner::Runner(const RunnerConfig& config, mllm::preprocessor::AutoTokenizer* tokenizer)
    : config_(config), tokenizer_(tokenizer) {}

bool Runner::load() {
  // init KV cache manager
  int32_t prompt_processor_ar_len = config_.ar_len;
  int32_t token_generator_ar_len = 1;

  if (prompt_processor_ar_len == config_.context_len) {
    config_.max_cache_len = config_.context_len;
  } else {
    config_.max_cache_len = config_.context_len - std::min(token_generator_ar_len, prompt_processor_ar_len);
  }
  config_.max_ar_len = std::max(token_generator_ar_len, prompt_processor_ar_len);
  kv_manager_ = std::make_unique<KVCacheManager<uint8_t>>(config_);

  auto backend = mllm::Context::instance().getBackend(mllm::kQNN);
  if (!backend) {
    MLLM_ERROR("QNN Backend not found");
    return false;
  }

  // init prompt processor(prefill)
  config_.use_int64_token = false;
  config_.sliding_window = config_.context_len;  // no sliding window for now

  prompt_processor_ = std::make_unique<PromptProcessor<uint8_t>>(kv_manager_.get(), config_);

  // init token generator(decode)
  // TODO: EOS IDs
  auto eos_ids = std::make_unique<std::unordered_set<uint64_t>>();
  eos_ids->insert(151643);
  eos_ids->insert(151645);

  token_generator_ = std::make_unique<TokenGenerator<uint8_t>>(tokenizer_, kv_manager_.get(), std::move(eos_ids), config_);

  kv_manager_->initCache(backend->allocator().get(), config_.ar_len);
  prompt_processor_->init_io();
  token_generator_->init_io();

  return true;
}

void Runner::generate(const Tensor& prompt_tokens, int32_t seq_len,
                      const std::function<void(const std::string&)>& token_callback) {
  MLLM_RT_ASSERT(prompt_tokens.rank() == 2 && prompt_tokens.dtype() == kInt64);

  int64_t start_pos = 0;

  std::vector<int64_t> prompt_tokens_i64;
  prompt_tokens_i64.reserve(prompt_tokens.shape()[1]);
  for (int i = 0; i < prompt_tokens.shape()[1]; i++) { prompt_tokens_i64.push_back(prompt_tokens.ptr<int64_t>()[i]); }

  int64_t next_token = prompt_processor_->prefill(prompt_tokens_i64, start_pos);

  if (token_callback) {
    std::wstring wstr = tokenizer_->detokenize(next_token);
    std::string str = mllm::preprocessor::wideString2Utf8String(wstr);
    token_callback(str);
  }

  // int64_t cur_pos = prompt_tokens.size(-1);

  // token_generator_->generate(prompt_tokens, cur_pos, seq_len, token_callback, false);
}

}  // namespace mllm::qnn::aot
