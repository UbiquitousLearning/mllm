// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <cstring>
#include <chrono>

#include "mllm/backends/qnn/aot_rt/QnnAOTRuntime.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"
#include <fmt/core.h>
#include <fmt/color.h>

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
                      const std::function<void(const std::string&)>& token_callback, bool perf) {
  MLLM_RT_ASSERT(prompt_tokens.rank() == 2 && prompt_tokens.dtype() == kInt64);

  int64_t start_pos = 0;

  std::vector<int64_t> prompt_tokens_i64;
  prompt_tokens_i64.reserve(prompt_tokens.shape()[1]);

  for (int i = 0; i < prompt_tokens.shape()[1]; i++) { prompt_tokens_i64.push_back(prompt_tokens.ptr<int64_t>()[i]); }

  // Measure prefill time
  std::chrono::high_resolution_clock::time_point prefill_start, prefill_end;
  if (perf) { prefill_start = std::chrono::high_resolution_clock::now(); }

  int64_t prefill_token_count = prompt_tokens_i64.size();
  int64_t next_token = prompt_processor_->prefill(prompt_tokens_i64, start_pos);
  prompt_tokens_i64.push_back(next_token);

  if (perf) { prefill_end = std::chrono::high_resolution_clock::now(); }

  if (token_callback) {
    std::wstring wstr = tokenizer_->detokenize(next_token);
    std::string str = mllm::preprocessor::wideString2Utf8String(wstr);
    token_callback(str);
  }

  int64_t cur_pos = prompt_tokens.size(-1);

  // Measure decode time
  std::chrono::high_resolution_clock::time_point decode_start, decode_end;
  if (perf) { decode_start = std::chrono::high_resolution_clock::now(); }

  int64_t generated_count = token_generator_->generate(prompt_tokens_i64, cur_pos, seq_len, token_callback, false);

  if (perf) {
    decode_end = std::chrono::high_resolution_clock::now();

    // Calculate durations in microseconds
    auto prefill_duration = std::chrono::duration_cast<std::chrono::microseconds>(prefill_end - prefill_start).count();
    auto decode_duration = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start).count();

    // Calculate TPS
    double prefill_tps = 0.0;
    double decode_tps = 0.0;

    if (prefill_duration > 0) { prefill_tps = (double)prefill_token_count / (prefill_duration / 1000000.0); }

    if (decode_duration > 0 && generated_count > 0) { decode_tps = (double)generated_count / (decode_duration / 1000000.0); }

    // Print performance summary
    fmt::print(fg(fmt::color::cyan), "\n{:=^50}\n", " Performance Summary ");
    fmt::print(fg(fmt::color::white), "{:<20}: ", "Prefill time");
    fmt::print(fg(fmt::color::yellow), "{:>10.2f} μs", (double)prefill_duration);
    if (prefill_tps > 0) { fmt::print(fg(fmt::color::white), " ({:>6.2f} tokens/s)", prefill_tps); }
    fmt::print("\n");

    fmt::print(fg(fmt::color::white), "{:<20}: ", "Decode time");
    fmt::print(fg(fmt::color::yellow), "{:>10.2f} μs", (double)decode_duration);
    if (decode_tps > 0) { fmt::print(fg(fmt::color::white), " ({:>6.2f} tokens/s)", decode_tps); }
    fmt::print("\n");

    fmt::print(fg(fmt::color::white), "{:<20}: ", "Prefill tokens");
    fmt::print(fg(fmt::color::green), "{:>10}\n", prefill_token_count);

    fmt::print(fg(fmt::color::white), "{:<20}: ", "Decode tokens");
    fmt::print(fg(fmt::color::green), "{:>10}\n", generated_count);
  }
}

}  // namespace mllm::qnn::aot
