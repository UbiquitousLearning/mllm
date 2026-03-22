// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Qwen3.5 Hybrid CPU+QNN Runtime — Phase 1
//
// Runs the Qwen3.5 0.8B model with:
//   - Embedding, GDN layers, final norm, lm_head on CPU
//   - 6 full attention layers on QNN (loaded from AOT context binary)
//
// The 18 GDN layers are sequential recurrent (not expressible as static QNN DAGs),
// so they stay on CPU. The 6 full attention layers are the compute-heavy parts that
// benefit from QNN HTP acceleration.
//
// Architecture per forward pass:
//   CPU: embed(token) -> h
//   CPU: GDN layers 0,1,2
//   QNN: full_attn_layer_3(h, kv_cache_0) -> h
//   CPU: GDN layers 4,5,6
//   QNN: full_attn_layer_7(h, kv_cache_1) -> h
//   ... (repeat pattern)
//   CPU: final_norm -> lm_head -> argmax

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/nn/Module.hpp>
#include <mllm/nn/Nn.hpp>
#include <mllm/nn/lmcache/StaticCache.hpp>
#include <mllm/models/qwen3_5/modeling_qwen3_5.hpp>
#include <mllm/models/qwen3_5/configuration_qwen3_5.hpp>
#include <mllm/models/qwen3_5/tokenization_qwen3_5.hpp>
#include <mllm/backends/qnn/aot_rt/QnnAOTRuntime.hpp>
#include <mllm/backends/qnn/aot_rt/QnnAOTModule.hpp>
#include <mllm/backends/qnn/aot_rt/KVCacheManager.hpp>

using mllm::Argparse;
using mllm::Tensor;
using mllm::AnyValue;
using namespace mllm::models::qwen3_5;  // NOLINT
using namespace mllm::qnn::aot;  // NOLINT

// Full attention layer indices for Qwen3.5 0.8B (every 4th layer starting from 3)
static constexpr int FULL_ATTN_LAYERS[] = {3, 7, 11, 15, 19, 23};
static constexpr int NUM_FULL_ATTN_LAYERS = 6;

// ==========================================================================
// Qwen3_5HybridModel
//
// A custom model class that constructs only the CPU components needed for
// hybrid execution. It loads weights from the same .mllm parameter file.
//
// Module hierarchy (matches weight names):
//   "model" -> "language_model" -> "embed_tokens"
//                               -> "layers.{i}"  (GDN layers only)
//                               -> "norm"
//   "lm_head_out"
//
// Full attention layers are NOT constructed here — they run on QNN.
// ==========================================================================

class Qwen3_5HybridModel final : public mllm::nn::Module {
 public:
  Qwen3_5HybridModel(const Qwen3_5Config& cfg)
      : mllm::nn::Module(""), cfg_(cfg) {
    // Build module hierarchy matching the weight name prefix:
    //   model.language_model.embed_tokens.weight
    //   model.language_model.layers.{i}.linear_attn.*
    //   model.language_model.norm.weight
    //   lm_head_out.weight (tied)
    lm_wrapper_ = reg<LMWrapper>("model", cfg);

    if (cfg.tie_word_embeddings) {
      lm_head_ = reg<mllm::nn::Linear>("lm_head_out", cfg.hidden_size, cfg.vocab_size,
                                        false, cfg.linear_impl_type);
    }

    // RoPE inverse frequency for position embedding computation
    inv_freq_ = makeRoPEInvFreq(cfg.rotary_dim(), cfg.rope_theta);
    registerBuffer("inv_freq", inv_freq_);

    // StaticCache for full attention layers (CPU fallback)
    kv_cache_ = mllm::nn::StaticCache(
        cfg.max_cache_length, cfg.num_hidden_layers,
        cfg.num_attention_heads, cfg.num_key_value_heads,
        cfg.head_dim, mllm::kFloat32, mllm::kCPU, false);
  }

  void resetStates() {
    lm_wrapper_.resetGDNStates(1);
    kv_cache_.clearCache();
    n_past_ = 0;
  }

  // Forward one chunk of tokens through the hybrid pipeline.
  // For CPU-only: all layers run on CPU.
  // For QNN hybrid: full attention layers dispatch to QNN (TODO: wire up).
  Tensor forwardChunk(const std::vector<int64_t>& token_ids) {
    int B = 1;
    int S = static_cast<int>(token_ids.size());

    // Input token IDs
    auto input_ids = Tensor::empty({B, S}, mllm::kInt64, mllm::kCPU).alloc();
    auto* id_ptr = input_ids.ptr<int64_t>();
    for (int i = 0; i < S; ++i) { id_ptr[i] = token_ids[i]; }

    // Position IDs
    auto position_ids = Tensor::empty({B, S}, mllm::kInt64, mllm::kCPU).alloc();
    auto* pos_ptr = position_ids.ptr<int64_t>();
    for (int i = 0; i < S; ++i) { pos_ptr[i] = n_past_ + i; }

    // RoPE embeddings (partial rotation)
    auto [sin_emb, cos_emb] = makeRotaryPosEmbedding(position_ids, inv_freq_, 1.0f);

    // Forward through model
    auto x = lm_wrapper_.forwardHybrid(input_ids, sin_emb, cos_emb, &kv_cache_);

    // Extract last token position
    x = x[{mllm::kAll, {S - 1}, mllm::kAll}];

    // LM head
    if (cfg_.tie_word_embeddings) {
      x = lm_head_(x);
    }

    n_past_ += S;
    return x;
  }

  int n_past_ = 0;

 private:
  // Inner wrapper matching "model.language_model.*" weight prefix
  class LMText final : public mllm::nn::Module {
   public:
    mllm::nn::Embedding embedding_;
    mllm::nn::RMSNorm norm_;

    // Full attention layers on CPU (used as fallback or for comparison)
    std::vector<Qwen3_5FullAttentionDecoder> full_attn_layers_;
    // GDN layers always on CPU
    std::vector<Qwen3_5GDNDecoder> gdn_layers_;

    // Dispatch table: for each global layer, which type and which index
    std::vector<int> layer_type_;      // 0 = full_attn, 1 = gdn
    std::vector<int> layer_dispatch_;  // index into respective vector

    LMText() = default;
    LMText(const std::string& name, const Qwen3_5Config& cfg) : mllm::nn::Module(name) {
      embedding_ = reg<mllm::nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);

      int gdn_count = 0;
      int attn_count = 0;
      for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        std::string layer_name = "layers." + std::to_string(i);
        if (cfg.isFullAttentionLayer(i)) {
          auto layer = reg<Qwen3_5FullAttentionDecoder>(layer_name, cfg);
          layer.self_attn_.layer_idx_ = i;
          full_attn_layers_.push_back(std::move(layer));
          layer_type_.push_back(0);
          layer_dispatch_.push_back(attn_count++);
        } else {
          auto layer = reg<Qwen3_5GDNDecoder>(layer_name, cfg);
          layer.linear_attn_.layer_idx_ = i;
          layer.linear_attn_.gdn_layer_idx_ = gdn_count;
          gdn_layers_.push_back(std::move(layer));
          layer_type_.push_back(1);
          layer_dispatch_.push_back(gdn_count++);
        }
      }

      norm_ = reg<mllm::nn::RMSNorm>("norm", cfg.rms_norm_eps, /*add_unit_offset=*/true);
    }

    void resetGDNStates(int batch_size) {
      for (auto& gdn : gdn_layers_) { gdn.linear_attn_.resetState(batch_size); }
    }

    // Hybrid forward: GDN on CPU, full attention on CPU (QNN dispatch TODO)
    Tensor forwardHybrid(
        const Tensor& input_ids,
        const Tensor& sin_emb,
        const Tensor& cos_emb,
        mllm::nn::StaticCache* kv_cache) {
      auto x = embedding_(input_ids);

      for (size_t i = 0; i < layer_type_.size(); ++i) {
        if (layer_type_[i] == 0) {
          // Full attention — CPU for now, QNN in Phase 1 integration
          x = full_attn_layers_[layer_dispatch_[i]](x, sin_emb, cos_emb, AnyValue(kv_cache))[0];
        } else {
          // GDN — always on CPU
          x = gdn_layers_[layer_dispatch_[i]](x, AnyValue(kv_cache))[0];
        }
      }

      x = norm_(x);
      return x;
    }

    std::vector<Tensor> forward(const std::vector<Tensor>& inputs,
                                const std::vector<AnyValue>& args) override {
      return {forwardHybrid(inputs[0], inputs[1], inputs[2],
                            args[0].get<mllm::nn::StaticCache*>())};
    }
  };

  // Wrapper matching "model.*" prefix
  class LMWrapper final : public mllm::nn::Module {
   public:
    LMText language_model_;

    LMWrapper() = default;
    LMWrapper(const std::string& name, const Qwen3_5Config& cfg) : mllm::nn::Module(name) {
      language_model_ = reg<LMText>("language_model", cfg);
    }

    void resetGDNStates(int batch_size) { language_model_.resetGDNStates(batch_size); }

    Tensor forwardHybrid(
        const Tensor& input_ids,
        const Tensor& sin_emb,
        const Tensor& cos_emb,
        mllm::nn::StaticCache* kv_cache) {
      return language_model_.forwardHybrid(input_ids, sin_emb, cos_emb, kv_cache);
    }

    std::vector<Tensor> forward(const std::vector<Tensor>& inputs,
                                const std::vector<AnyValue>& args) override {
      return language_model_(inputs[0], inputs[1], inputs[2], args[0]);
    }
  };

  const Qwen3_5Config& cfg_;
  LMWrapper lm_wrapper_;
  mllm::nn::Linear lm_head_;
  Tensor inv_freq_;
  mllm::nn::StaticCache kv_cache_;
};

// ==========================================================================
// HybridQwen3_5Runner
//
// Manages generation using the hybrid model.
// ==========================================================================

class HybridQwen3_5Runner {
 public:
  HybridQwen3_5Runner(
      const Qwen3_5Config& cfg,
      const std::string& model_path,
      const std::string& tokenizer_path,
      bool use_qnn)
      : cfg_(cfg),
        tokenizer_(tokenizer_path),
        use_qnn_(use_qnn) {
    model_ = std::make_unique<Qwen3_5HybridModel>(cfg);
    auto params = mllm::load(model_path, mllm::ModelFileVersion::kV2);
    model_->load(params);

    if (use_qnn_) {
      initQnnModules();
    }
  }

  void generate(const std::string& prompt, int max_tokens = 256) {
    model_->resetStates();

    auto inputs = tokenizer_.convertMessage({.prompt = prompt});
    auto& sequence = inputs["sequence"];

    auto token_ids = extractTokenIds(sequence);
    int prompt_len = static_cast<int>(token_ids.size());

    fmt::print("\nResponse: ");

    // --- Prefill phase ---
    // Process prompt in chunks
    int prefill_chunk = 32;
    int64_t next_token = 0;
    for (int pos = 0; pos < prompt_len; pos += prefill_chunk) {
      int chunk_len = std::min(prefill_chunk, prompt_len - pos);
      std::vector<int64_t> chunk(token_ids.begin() + pos, token_ids.begin() + pos + chunk_len);
      auto logits = model_->forwardChunk(chunk);

      if (pos + prefill_chunk >= prompt_len) {
        next_token = sampleGreedy(logits);
        token_ids.push_back(next_token);
        printToken(next_token);
        if (isEos(next_token)) goto done;
      }
    }

    // --- Decode phase ---
    for (int i = 0; i < max_tokens - 1; ++i) {
      auto logits = model_->forwardChunk({token_ids.back()});
      next_token = sampleGreedy(logits);
      token_ids.push_back(next_token);
      printToken(next_token);
      if (isEos(next_token)) break;
    }

  done:
    fmt::print("\n");
  }

 private:
  void initQnnModules() {
    // Create QNN modules for each full attention layer:
    //   Prefill: graph "model.attn{i}.s32"
    //   Decode:  graph "model.attn{i}.s1"
    for (int i = 0; i < NUM_FULL_ATTN_LAYERS; ++i) {
      attn_prefill_.push_back(
          std::make_unique<QnnAOTModule>("model.attn" + std::to_string(i) + ".s32"));
      attn_decode_.push_back(
          std::make_unique<QnnAOTModule>("model.attn" + std::to_string(i) + ".s1"));
    }

    // KV cache for 6 attention layers (quantized uint8)
    QnnAOTConfig kv_cfg;
    kv_cfg.num_layers = NUM_FULL_ATTN_LAYERS;
    kv_cfg.num_heads = cfg_.num_key_value_heads;
    kv_cfg.head_dim = cfg_.head_dim;
    kv_cfg.vocab_size = cfg_.vocab_size;
    kv_cfg.context_len = cfg_.max_cache_length;
    kv_cfg.ar_len = 32;
    kv_manager_ = std::make_unique<KVCacheManager<uint8_t>>(kv_cfg);

    fmt::print("QNN modules initialized: {} attention layers x 2 (prefill+decode)\n",
               NUM_FULL_ATTN_LAYERS);

    // TODO(Phase 1 integration): Wire QNN modules into
    // Qwen3_5HybridModel::LMText::forwardHybrid() to replace CPU attention.
    // This requires:
    // 1. Preparing QNN input tensors (hidden_states, sin/cos, mask, past_key/value)
    // 2. Calling QnnAOTModule forward
    // 3. Extracting output hidden_states + present KV
    // 4. Updating KVCacheManager with new KV entries
    // Cannot be completed until QNN SDK is available for on-device testing.
  }

  std::vector<int64_t> extractTokenIds(const Tensor& t) {
    std::vector<int64_t> ids;
    auto* ptr = t.ptr<int64_t>();
    int len = t.shape()[1];
    for (int i = 0; i < len; ++i) { ids.push_back(ptr[i]); }
    return ids;
  }

  int64_t sampleGreedy(const Tensor& logits) {
    auto* ptr = logits.ptr<float>();
    int vocab_size = logits.shape()[-1];
    int64_t best_id = 0;
    float best_val = ptr[0];
    for (int i = 1; i < vocab_size; ++i) {
      if (ptr[i] > best_val) {
        best_val = ptr[i];
        best_id = i;
      }
    }
    return best_id;
  }

  void printToken(int64_t token_id) {
    std::wcout << tokenizer_.detokenize(token_id) << std::flush;
  }

  bool isEos(int64_t token_id) {
    return token_id == cfg_.eos_token_id
        || token_id == cfg_.im_end_token_id
        || token_id == cfg_.end_of_text_token_id;
  }

  const Qwen3_5Config& cfg_;
  Qwen3_5Tokenizer tokenizer_;
  std::unique_ptr<Qwen3_5HybridModel> model_;
  bool use_qnn_;

  // QNN modules (used when use_qnn_ == true)
  std::vector<std::unique_ptr<QnnAOTModule>> attn_prefill_;
  std::vector<std::unique_ptr<QnnAOTModule>> attn_decode_;
  std::unique_ptr<KVCacheManager<uint8_t>> kv_manager_;
};

// ==========================================================================
// Main entry point
// ==========================================================================

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model")
      .help("Model path (.mllm file with pre-baked weights)").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer")
      .help("Tokenizer directory").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config")
      .help("Config path (config_mllm.json)").required(true);
  auto& qnn_context = Argparse::add<std::string>("--qnn_context")
      .help("QNN context binary path (omit for CPU-only mode)").def("");

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    mllm::shutdownContext();
    return 0;
  }

  auto cfg = Qwen3_5Config(config_path.get());
  bool use_qnn = !qnn_context.get().empty();

  if (use_qnn) {
    mllm::initQnnBackend(qnn_context.get());
    fmt::print("QNN backend initialized with context: {}\n", qnn_context.get());
  } else {
    fmt::print("Running in CPU-only mode (no QNN context binary provided)\n");
  }

  fmt::print("Qwen3.5 0.8B: {} layers ({} full attention on {}, {} GDN on CPU)\n",
             cfg.num_hidden_layers,
             cfg.numFullAttentionLayers(), use_qnn ? "QNN" : "CPU",
             cfg.numGDNLayers());

  HybridQwen3_5Runner runner(cfg, model_path.get(), tokenizer_path.get(), use_qnn);

  fmt::print("\n{:*^60}\n", " Qwen3.5 Hybrid Interactive CLI ");
  fmt::print("Enter 'exit' or 'quit' to end the session\n\n");

  while (true) {
    std::string prompt_text;
    fmt::print("Prompt: ");
    std::getline(std::cin, prompt_text);

    if (prompt_text == "exit" || prompt_text == "quit" || std::cin.eof()) { break; }
    if (prompt_text.empty()) { continue; }

    try {
      runner.generate(prompt_text);
    } catch (const std::exception& e) {
      fmt::print("\nError: {}\n", e.what());
    }
  }

  mllm::print("\n");
  mllm::memoryReport();
});
