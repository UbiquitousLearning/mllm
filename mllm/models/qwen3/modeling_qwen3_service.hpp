// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <span>
#include <memory>
#include <ranges>
#include <algorithm>
#include <filesystem>

#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/preprocessor/tokenizers/Unicode.hpp"
#include "mllm/models/qwen3/tokenization_qwen3.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"

// Service related.
#include "mllm/engine/service/Session.hpp"
#include "mllm/engine/prefix_cache/Cache.hpp"

namespace mllm::models::qwen3 {

inline auto makeRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) { inv_freq_ptr[i] = 1.0 / std::pow(rope_theta, 2.0 * i / output_dim); }
  return inv_freq;
}

inline auto makeRotaryPosEmbedding(Tensor& position_ids, const Tensor& inv_freq,
                                   float attention_scaling = 1.0f) -> std::pair<Tensor, Tensor> {
  auto batch_size = position_ids.shape()[0];
  auto seq_len = position_ids.shape()[1];
  auto inv_freq_len = inv_freq.shape()[0];
  auto dim = inv_freq_len * 2;

  // Create freqs tensor: position_ids @ inv_freq
  auto freqs = Tensor::empty({batch_size, seq_len, inv_freq_len}, kFloat32, kCPU).alloc();
  auto freqs_ptr = freqs.ptr<float>();
  auto position_ids_ptr = position_ids.ptr<int64_t>();
  auto inv_freq_ptr = inv_freq.ptr<float>();

  // Compute freqs = position_ids[:, :, None] @ inv_freq[None, :]
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      auto pos = position_ids_ptr[b * seq_len + s];
      for (int d = 0; d < inv_freq_len; ++d) {
        freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d] = static_cast<float>(pos) * inv_freq_ptr[d];
      }
    }
  }

  // Create sin and cos tensors with shape [batch_size, seq_len, dim]
  auto sin_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto cos_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto sin_ptr = sin_emb.ptr<float>();
  auto cos_ptr = cos_emb.ptr<float>();

  // Compute sin and cos embeddings: emb = [freqs, freqs]
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      for (int d = 0; d < inv_freq_len; ++d) {
        auto freq = freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d];
        auto sin_val = std::sin(freq) * attention_scaling;
        auto cos_val = std::cos(freq) * attention_scaling;

        // Store the same values in both halves: [freqs, freqs]
        sin_ptr[b * seq_len * dim + s * dim + d] = sin_val;
        sin_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = sin_val;
        cos_ptr[b * seq_len * dim + s * dim + d] = cos_val;
        cos_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = cos_val;
      }
    }
  }

  return {sin_emb, cos_emb};
}

class Qwen3MLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

 public:
  Qwen3MLP() = default;
  Qwen3MLP(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    gate_proj_ = reg<nn::Linear>("gate_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    silu_ = reg<nn::SiLU>("act");
    up_proj_ = reg<nn::Linear>("up_proj", cfg.hidden_size, cfg.intermediate_size, false, cfg.linear_impl_type);
    down_proj_ = reg<nn::Linear>("down_proj", cfg.intermediate_size, cfg.hidden_size, false, cfg.linear_impl_type);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = gate_proj_(inputs[0]);
    x = silu_(x);
    auto y = up_proj_(inputs[0]);
    x = x * y;
    x = down_proj_(x);
    return {x};
  }
};

class Qwen3Attention final : public nn::Module {
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear o_proj_;
  nn::RMSNorm rms_norm_q_;
  nn::RMSNorm rms_norm_k_;
  nn::RoPE q_rope_;
  nn::RoPE k_rope_;
  nn::RadixAttn attn_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;

 public:
  Qwen3Attention() = default;

  Qwen3Attention(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;

    // clang-format off
    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, cfg.attention_bias, cfg.linear_impl_type);
    k_proj_ = reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
    v_proj_ = reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, cfg.attention_bias, cfg.linear_impl_type);
    o_proj_ = reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, cfg.attention_bias, cfg.linear_impl_type);
    // clang-format on

    rms_norm_q_ = reg<nn::RMSNorm>("q_norm", cfg.rms_norm_eps);
    rms_norm_k_ = reg<nn::RMSNorm>("k_norm", cfg.rms_norm_eps);

    q_rope_ = reg<nn::RoPE>("q_rope", cfg.rope_theta, cfg.max_position_embeddings, aops::RoPEOpOptionsInputType::kBSHD);
    k_rope_ = reg<nn::RoPE>("k_rope", cfg.rope_theta, cfg.max_position_embeddings, aops::RoPEOpOptionsInputType::kBSHD);

    attn_ = reg<nn::RadixAttn>("attn", num_attention_heads_, num_key_value_heads_);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto k_cache_addr = args[0].get<std::vector<std::vector<prefix_cache::vp_addr_t>>*>();
    auto v_cache_addr = args[1].get<std::vector<std::vector<prefix_cache::vp_addr_t>>*>();
    auto prefix_cache_context = args[2].get<prefix_cache::Cache*>();

    // [B, S, H * D]
    auto query_states = q_proj_(x);
    auto key_states = k_proj_(x);
    auto value_states = v_proj_(x);

    int B = inputs[0].shape()[0];
    int S = inputs[0].shape()[1];

    // [B, S, H, D]
    query_states = query_states.view({B, S, num_attention_heads_, head_dim_});
    key_states = key_states.view({B, S, num_key_value_heads_, head_dim_});
    value_states = value_states.view({B, S, num_key_value_heads_, head_dim_});

    // [B, S, H, D]
    query_states = rms_norm_q_(query_states);
    key_states = rms_norm_k_(key_states);

    // Different from original [B, H, S, D] rope.
    // [B, S, H, D]
    query_states = q_rope_(query_states, llm_embedding_sin, llm_embedding_cos);
    key_states = k_rope_(key_states, llm_embedding_sin, llm_embedding_cos);

    // FIXME: Think, if rope before cache is ok?

    // Acquire cache
    std::vector<prefix_cache::vp_addr_t> k_addr_wait_for_promote;
    std::vector<prefix_cache::vp_addr_t> v_addr_wait_for_promote;
    for (int s_idx = 0; s_idx < S; ++s_idx) {
      k_addr_wait_for_promote.push_back(prefix_cache_context->alloc(kCPU));
      v_addr_wait_for_promote.push_back(prefix_cache_context->alloc(kCPU));
    }

    // Prepare indicies cache. sizeof(char*) == 8 == sizeof(int64_t)
    std::vector<char*> k_phy_addr_wait_for_promote;
    std::vector<char*> v_phy_addr_wait_for_promote;
    for (int s_idx = 0; s_idx < S; ++s_idx) {
      k_phy_addr_wait_for_promote.push_back(prefix_cache_context->physicalAddr(k_addr_wait_for_promote[s_idx]));
      v_phy_addr_wait_for_promote.push_back(prefix_cache_context->physicalAddr(v_addr_wait_for_promote[s_idx]));
    }
    auto k_wait_for_promote = Tensor::refVectorData(k_phy_addr_wait_for_promote, {S}, kInt64, kCPU);
    auto v_wait_for_promote = Tensor::refVectorData(v_phy_addr_wait_for_promote, {S}, kInt64, kCPU);

    // Copy key_states and value_states to cache
    nn::functional::scatter2Shards(key_states, k_wait_for_promote, 1);
    nn::functional::scatter2Shards(value_states, v_wait_for_promote, 1);

    // Gather all cache to indicies tensor
    {
      auto& dst = (*k_cache_addr)[layer_idx_];
      dst.insert(dst.end(), k_addr_wait_for_promote.begin(), k_addr_wait_for_promote.end());
    }
    {
      auto& dst = (*v_cache_addr)[layer_idx_];
      dst.insert(dst.end(), v_addr_wait_for_promote.begin(), v_addr_wait_for_promote.end());
    }
    std::vector<char*> k_phy_cache_indicies;
    std::vector<char*> v_phy_cache_indicies;
    int32_t kv_cache_len = (*k_cache_addr)[layer_idx_].size();
    k_phy_cache_indicies.reserve(kv_cache_len);
    v_phy_cache_indicies.reserve(kv_cache_len);
    for (int i = 0; i < kv_cache_len; ++i) {
      k_phy_cache_indicies.push_back(prefix_cache_context->physicalAddr((*k_cache_addr)[layer_idx_][i]));
      v_phy_cache_indicies.push_back(prefix_cache_context->physicalAddr((*v_cache_addr)[layer_idx_][i]));
    }
    auto k_cache = Tensor::refVectorData(k_phy_cache_indicies, {kv_cache_len}, kInt64, kCPU);
    auto v_cache = Tensor::refVectorData(v_phy_cache_indicies, {kv_cache_len}, kInt64, kCPU);

    // Do Radix Attention
    // output is [B, S, H, D]
    auto output = attn_(query_states, k_cache, v_cache);
    output = output.view({B, S, num_attention_heads_ * head_dim_});
    output = o_proj_(output);

    return {output};
  }

  int layer_idx_;
};

class Qwen3Decoder final : public nn::Module {
 public:
  Qwen3Attention self_attn_;
  Qwen3MLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  Qwen3Decoder() = default;

  Qwen3Decoder(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    self_attn_ = reg<Qwen3Attention>("self_attn", cfg);
    mlp_ = reg<Qwen3MLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& k_cache_addr = args[0];
    auto& v_cache_addr = args[1];
    auto& prefix_cache_context = args[2];

    auto x = input_layer_norm_(inputs[0]);
    x = self_attn_(x, llm_embedding_sin, llm_embedding_cos, k_cache_addr, v_cache_addr, prefix_cache_context)[0];
    auto tmp = x + inputs[0];
    x = post_attention_layer_norm_(tmp);
    x = mlp_(x)[0];
    x = x + tmp;
    return {x};
  }
};

class Qwen3Text final : public nn::Module {
  nn::ModuleList<Qwen3Decoder> decode_blocks_;
  nn::RMSNorm norm_;
  nn::Embedding embedding_;

 public:
  Qwen3Text() = default;

  Qwen3Text(const std::string& name, const Qwen3Config& cfg) : nn::Module(name) {
    decode_blocks_ = reg<nn::ModuleList<Qwen3Decoder>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& blocks = decode_blocks_.list();

    // X is already embedded
    auto x = embedding_(inputs[0]);

    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto& k_cache_addr = args[0];
    auto& v_cache_addr = args[1];
    auto& prefix_cache_context = args[2];

    for (auto& block : blocks) {
      x = block(x, llm_embedding_sin, llm_embedding_cos, k_cache_addr, v_cache_addr, prefix_cache_context)[0];
    }

    x = norm_(x);

    return {x};
  }
};

class Qwen3ForCausalLM : public ARGeneration, public nn::Module {
 public:
  explicit Qwen3ForCausalLM(const Qwen3Config& cfg) : cfg(cfg) {
    eos_token_id_ = cfg.end_of_text_token_id;
    max_length_ = cfg.max_cache_length;
    tie_word_embeddings_ = cfg.tie_word_embeddings;

    llm = reg<Qwen3Text>("model", cfg);

    // Qwen3 0.6B's lm_head is tied with embed_tokens. But ModelScope's official weights separate them.
    if (cfg.tie_word_embeddings) {
      // NOTE:
      // model.lm_head.weight is quantization weights of model.embed_tokens.weight
      lm_head_ = reg<nn::Linear>("lm_head_out", cfg.hidden_size, cfg.vocab_size, false, cfg.linear_impl_type);
    }

    // Init inv freq
    auto inv = makeRoPEInvFreq(cfg.head_dim, cfg.rope_theta);
    registerBuffer("inv_freq", inv);
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    auto sequence = input.at("sequence");

    // Generate position_ids for the current sequence
    auto batch_size = sequence.shape()[0];
    auto seq_len = sequence.shape()[1];

    Tensor position_ids = Tensor::nil();
    if (input.count("position_ids")) {
      // Use existing position_ids for decode phase
      position_ids = input.at("position_ids");

      // For decode phase, increment the last position
      if (seq_len == 1) {
        auto last_pos = *position_ids.offsettedPtr<int64_t>({0, position_ids.shape()[1] - 1});
        position_ids = Tensor::empty({batch_size, 1}, kInt64, kCPU).alloc();
        *position_ids.offsettedPtr<int64_t>({0, 0}) = last_pos + 1;
      }
    } else {
      // NOTE: Service Session should not go into this branch !!!
      MLLM_RT_ASSERT(false);

      // Generate position_ids for prefill phase
      position_ids = Tensor::empty({batch_size, seq_len}, kInt64, kCPU).alloc();
      auto position_ids_ptr = position_ids.ptr<int64_t>();
      for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) { position_ids_ptr[b * seq_len + s] = s; }
      }
    }

    // Generate RoPE embeddings using the inv_freq buffer
    auto [llm_embedding_sin, llm_embedding_cos] = makeRotaryPosEmbedding(position_ids, getBuffer("inv_freq"), 1.0f);

    sequence = llm(sequence, llm_embedding_sin, llm_embedding_cos, args.at("k_cache_addrs"), args.at("v_cache_addrs"),
                   args.at("prefix_cache_context"))[0];

    // clip x to one seq length
    {
      auto S = sequence.shape()[1];
      sequence = sequence[{kAll, {S - 1}, kAll}];
    }
    if (tie_word_embeddings_) { sequence = lm_head_(sequence); }

    return {
        {"sequence", sequence},
        {"position_ids", position_ids},
    };
  }

  const Qwen3Config cfg;

 private:
  Qwen3Text llm;
  nn::Linear lm_head_;
  bool tie_word_embeddings_;
};

class Qwen3Session final : public ::mllm::service::Session {
 public:
  Qwen3Session() = default;

  std::size_t findThinkStartToken(const std::vector<int64_t>& output_ids) {
    auto it = std::find(output_ids.begin(), output_ids.end(), model_->cfg.thinking_start_token_id);
    return std::distance(output_ids.begin(), it);
  }

  void streamGenerate(const nlohmann::json& request,
                      const std::function<void(const nlohmann::json&, bool)>& callback) override {
    const auto& messages = request["messages"];
    auto inputs = applyChatTemplate(messages, {}, true, request.value("enable_thinking", false));

    auto full_seq_idx = tokenizer_->convert2Ids(tokenizer_->tokenize(inputs)).toVector<int64_t>();
    ARGenerationArgs args;
    ARGenerationOutputPast input;

    // Search in the radix cache. Find the tokens we really need to compute.
    auto prefix_cache_result = cache_->find(full_seq_idx);
    fmt::print("Reuse Tokens: {}, compute budget: {} tokens\n", prefix_cache_result.matched_length,
               full_seq_idx.size() - prefix_cache_result.matched_length);
    std::span<int64_t> reduced_seq_idx(full_seq_idx.data() + prefix_cache_result.matched_length,
                                       full_seq_idx.size() - prefix_cache_result.matched_length);
    std::vector<int64_t> position_ids;
    {
      auto start = prefix_cache_result.matched_length;
      auto end = full_seq_idx.size();
      position_ids.reserve(end - start);
      std::ranges::copy(std::views::iota(static_cast<int64_t>(start), static_cast<int64_t>(end)),
                        std::back_inserter(position_ids));
    }
    MLLM_RT_ASSERT_EQ(reduced_seq_idx.size(), position_ids.size());
    input["sequence"] = Tensor::fromVector(reduced_seq_idx, {1, (int32_t)reduced_seq_idx.size()}, kInt64, kCPU);
    input["position_ids"] = Tensor::fromVector(position_ids, {1, (int32_t)position_ids.size()}, kInt64, kCPU);

    // Setup session context
    k_cache_addrs_ = prefix_cache_result.k_cache_addresses;
    v_cache_addrs_ = prefix_cache_result.v_cache_addresses;
    args["k_cache_addrs"] = &k_cache_addrs_;
    args["v_cache_addrs"] = &v_cache_addrs_;
    args["prefix_cache_context"] = cache_.get();

    // Has temperature, top_k, top_p, max_length, do_sample.
    args["temperature"] = request.value("temperature", 1.0f);
    args["top_k"] = request.value("top_k", 0);
    args["top_p"] = request.value("top_p", 0.0f);
    args["max_length"] = request.value("max_length", 1024);
    args["do_sample"] = request.value("do_sample", false);

    // Iteration start
    int64_t package_cnt = 0;
    model_->streamGenerate(input, args, [this, &request, &full_seq_idx, &package_cnt, &callback](int64_t idx) {
      bool finished = false;
      std::string ret_token;
      if (idx == model_->cfg.eos_token_id) {
        finished = true;
        ret_token = "";
      } else {
        finished = false;
        ret_token = preprocessor::wideString2Utf8String(tokenizer_->detokenize(idx));

        // Update full_seq_idx to include the new token for Radix Tree to use.
        full_seq_idx.push_back(idx);
      }

      // Callback will send this json to the response pool for user to consume.
      callback(ret_token, finished);

      package_cnt++;
    });
    // Callback a finish token
    callback("", true);

    // Post process full_seq_idx and k_cache_addrs_/v_cache_addrs_. Only none thinking budget should be insert in radix tree.
    //
    // NOTE: We will drop everything after the thinking_start_token_idx(include it).
    // Suppose: Only one <think> token in the sequence.
    //
    // e.g.:
    // <|im_start|>user
    // hello<|im_end|>
    // <|im_start|>assistant
    // <think>
    //
    // </think>
    // hello!
    // <|endoftext|>
    //
    // In radic tree, we will only save:
    // <|im_start|>user
    // hello<|im_end|>
    // <|im_start|>assistant
    //
    // Explain: That because Qwen3 and other CoT model will remove thinking budget, which means the answer "hello"'s rope is
    // changed in 2ed turn.
    auto thinking_end_token_idx = findThinkStartToken(full_seq_idx);
    full_seq_idx.resize(thinking_end_token_idx);
    for (auto& k_vec : k_cache_addrs_) k_vec.resize(thinking_end_token_idx);
    for (auto& v_vec : v_cache_addrs_) v_vec.resize(thinking_end_token_idx);

    // Insert generated tokens to the cache.
    cache_->promote(full_seq_idx, k_cache_addrs_, v_cache_addrs_);

    // Cleanup session Context
    k_cache_addrs_ = {};
    v_cache_addrs_ = {};
  }

  void fromPreTrain(const std::string& model_path) override {
    namespace fs = std::filesystem;
    fs::path root = fs::path(model_path).lexically_normal();
    fs::path config_file = root / "config.json";
    fs::path model_file = root / "model.mllm";
    fs::path tokenizer_file = root / "tokenizer.json";
    if (!fs::exists(config_file)) throw std::runtime_error(config_file.string() + " not found");
    if (!fs::exists(model_file)) throw std::runtime_error(model_file.string() + " not found");
    if (!fs::exists(tokenizer_file)) throw std::runtime_error(tokenizer_file.string() + " not found");

    auto cfg = Qwen3Config(config_file.string());
    model_ = std::make_shared<Qwen3ForCausalLM>(cfg);
    model_->load(load(model_file.string(), ModelFileVersion::kV2));
    tokenizer_ = std::make_shared<Qwen3Tokenizer>(tokenizer_file.string());

    cache_ = std::make_shared<prefix_cache::Cache>(prefix_cache::CacheOptions{
        .radix_tree_options = {.enable_lru_eviction = false,
                               .eviction_threshold = 0.9f,
                               .enable_path_compression = false,
                               .min_compression_length = 2,
                               .transformer_blocks_num = cfg.num_hidden_layers},
        .allocator_options = {// Normal things.
                              .per_k_token_ele = static_cast<size_t>(cfg.head_dim * cfg.num_key_value_heads),
                              .per_v_token_ele = static_cast<size_t>(cfg.head_dim * cfg.num_key_value_heads),
                              .k_dtype = mllm::kFloat32,
                              .v_dtype = mllm::kFloat32,

                              // CUDA things.
                              .enable_cuda = false,
                              .cuda_mem_base = 0x100000,

                              // CPU things.
                              .enable_cpu_hierarchy_memory = true,
                              .zen_fs_options = {
                                  .record = false,
                                  .working_dir = ".",
                                  .blob_bits_size = 20,
                                  .page_bits = 7,
                                  .lane_bits = 5,
                                  .per_k_token_ele = static_cast<size_t>(cfg.head_dim * cfg.num_key_value_heads),
                                  .per_v_token_ele = static_cast<size_t>(cfg.head_dim * cfg.num_key_value_heads),
                                  .k_dtype = mllm::kFloat32,
                                  .v_dtype = mllm::kFloat32,
                                  .mmap_type = mllm::prefix_cache::ZenFSBlobMMapType::kAnonymous,
                              }}});
  }

  std::string ltrim(const std::string& s) {
    size_t start = s.find_first_not_of(" \n\r\t\f\v");
    return (start == std::string::npos) ? "" : s.substr(start);
  }

  std::string rtrim(const std::string& s) {
    size_t end = s.find_last_not_of(" \n\r\t\f\v");
    return (end == std::string::npos) ? "" : s.substr(0, end + 1);
  }

  std::string trim(const std::string& s) { return rtrim(ltrim(s)); }

  std::string applyChatTemplate(const json& messages, const std::vector<json>& tools = {}, bool add_generation_prompt = true,
                                bool enable_thinking = true, const std::string& bos_token = "",
                                const std::string& eos_token = "<|im_end|>") {
    std::ostringstream oss;

    if (!tools.empty()) {
      oss << "<|im_start|>system\n";
      if (!messages.empty() && messages[0].value("role", "") == "system") { oss << messages[0].value("content", "") << "\n\n"; }
      oss << "# Tools\n\nYou may call one or more functions to assist with the user query.\n";
      oss << "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
      for (const auto& tool : tools) { oss << "\n" << tool.dump(); }
      oss << "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
             "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
             "<args-json-object>}\n</tool_call><|im_end|>\n";
    } else {
      if (!messages.empty() && messages[0].value("role", "") == "system") {
        oss << "<|im_start|>system\n" << messages[0].value("content", "") << "<|im_end|>\n";
      }
    }

    size_t last_query_index = messages.empty() ? 0 : messages.size() - 1;
    bool found_last_query = false;
    if (!messages.empty()) {
      for (int i = messages.size() - 1; i >= 0; --i) {
        const auto& msg = messages[i];
        if (msg.value("role", "") == "user" && msg.contains("content") && msg["content"].is_string()) {
          std::string content_str = msg["content"].get<std::string>();
          if (!(content_str.starts_with("<tool_response>")
                && content_str.find("</tool_response>") == content_str.length() - std::string("</tool_response>").length())) {
            last_query_index = i;
            found_last_query = true;
            break;
          }
        }
      }
    }
    if (messages.empty()) { found_last_query = false; }

    for (size_t i = 0; i < messages.size(); ++i) {
      const auto& message = messages[i];
      std::string role = message.value("role", "");
      std::string content;
      if (message.contains("content") && message["content"].is_string()) { content = message["content"].get<std::string>(); }

      if (role == "user" || (role == "system" && i > 0)) {
        oss << "<|im_start|>" << role << "\n" << content << "<|im_end|>\n";
      } else if (role == "assistant") {
        std::string reasoning_content;
        if (message.contains("reasoning_content") && message["reasoning_content"].is_string()) {
          reasoning_content = message["reasoning_content"].get<std::string>();
        } else {
          auto think_end_pos = content.find("</think>");
          if (think_end_pos != std::string::npos) {
            auto think_start_pos = content.rfind("<think>", think_end_pos);
            if (think_start_pos != std::string::npos) {
              reasoning_content = content.substr(think_start_pos + 7, think_end_pos - (think_start_pos + 7));
              content = content.substr(think_end_pos + 8);
            }
          }
        }

        oss << "<|im_start|>" << role << "\n";
        if (found_last_query && i > last_query_index) {
          if ((i == messages.size() - 1) || !reasoning_content.empty()) {
            oss << "<think>\n" << trim(reasoning_content) << "\n</think>\n\n" << ltrim(content);
          } else {
            oss << content;
          }
        } else {
          oss << content;
        }

        if (message.contains("tool_calls")) {
          bool is_first_tool = true;
          for (const auto& tool_call_item : message["tool_calls"]) {
            if ((is_first_tool && !content.empty()) || !is_first_tool) { oss << "\n"; }
            is_first_tool = false;

            const json* tool_call_ptr = &tool_call_item;
            if (tool_call_item.contains("function")) { tool_call_ptr = &tool_call_item["function"]; }
            const json& tool_call = *tool_call_ptr;

            oss << "<tool_call>\n{\"name\": \"" << tool_call.value("name", "") << R"(", "arguments": )";
            const auto& args = tool_call["arguments"];
            if (args.is_string()) {
              oss << args.get<std::string>();
            } else {
              oss << args.dump();
            }
            oss << "}\n</tool_call>";
          }
        }
        oss << "<|im_end|>\n";

      } else if (role == "tool") {
        if (i == 0 || messages[i - 1].value("role", "") != "tool") { oss << "<|im_start|>user"; }
        oss << "\n<tool_response>\n" << content << "\n</tool_response>";
        if (i == messages.size() - 1 || messages[i + 1].value("role", "") != "tool") { oss << "<|im_end|>\n"; }
      }
    }

    if (add_generation_prompt) {
      oss << "<|im_start|>assistant\n";
      if (!enable_thinking) { oss << "<think>\n\n</think>\n\n"; }
    }

    return oss.str();
  }

 private:
  // States
  std::vector<std::vector<prefix_cache::vp_addr_t>> k_cache_addrs_;
  std::vector<std::vector<prefix_cache::vp_addr_t>> v_cache_addrs_;

  // Owned  data
  std::shared_ptr<Qwen3ForCausalLM> model_;
  std::shared_ptr<Qwen3Tokenizer> tokenizer_;
  std::shared_ptr<prefix_cache::Cache> cache_;
};

}  // namespace mllm::models::qwen3
