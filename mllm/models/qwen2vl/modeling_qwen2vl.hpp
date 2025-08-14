// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <array>

#include "mllm/mllm.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"
#include "mllm/models/qwen2vl/output_past_qwen2vl.hpp"
#include "mllm/models/qwen2vl/configuration_qwen2vl.hpp"
#include "mllm/utils/Enumerate.hpp"

namespace mllm::models::qwen2vl {

class PatchEmbed final : public nn::Module {
  int32_t in_chans_;
  int32_t embed_dim_;
  int32_t patch_size_;
  int32_t temporal_patch_size_;

  nn::Conv3D proj_;

 public:
  PatchEmbed() = default;

  inline PatchEmbed(const std::string& name, const Qwen2VLConfig& cfg) : nn::Module(name) {
    in_chans_ = cfg.visual_in_chans;
    embed_dim_ = cfg.visual_embed_dim;
    patch_size_ = cfg.visual_patch_size;
    temporal_patch_size_ = cfg.visual_temporal_patch_size;

    proj_ = reg<nn::Conv3D>("proj", cfg.visual_in_chans, cfg.visual_embed_dim,
                            std::vector<int32_t>{cfg.visual_temporal_patch_size, cfg.visual_patch_size, cfg.visual_patch_size},
                            std::vector<int32_t>{cfg.visual_temporal_patch_size, cfg.visual_patch_size, cfg.visual_patch_size},
                            false);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];

    // [batch_size(1), in_channel(3), temporal_patch_size(2), patch_size(14), patch_size(14)]
    hidden_states = hidden_states.view({-1, in_chans_, temporal_patch_size_, patch_size_, patch_size_});
    hidden_states = proj_(hidden_states).view({-1, embed_dim_});

    return {hidden_states};
  }
};

class PatchMerger final : public nn::Module {
  int32_t hidden_size_;
  int32_t spatial_merge_size_;
  int32_t context_dim_;

  nn::LayerNorm ln_q_;
  nn::Linear mlp_0_;
  nn::Linear mlp_2_;
  nn::GELU mlp_gelu_;

 public:
  PatchMerger() = default;

  inline PatchMerger(const std::string& name, const Qwen2VLConfig& cfg) : nn::Module(name) {
    context_dim_ = cfg.visual_embed_dim;
    spatial_merge_size_ = cfg.visual_spatial_merge_size;
    hidden_size_ = context_dim_ * spatial_merge_size_ * spatial_merge_size_;

    ln_q_ = reg<nn::LayerNorm>("ln_q", std::vector<int32_t>{context_dim_}, true, true, 1e-6);
    mlp_0_ = reg<nn::Linear>("mlp.0", hidden_size_, hidden_size_, true);
    mlp_gelu_ = reg<nn::GELU>("mlp.gelu");
    mlp_2_ = reg<nn::Linear>("mlp.2", hidden_size_, cfg.hidden_size, true);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto o = ln_q_(inputs[0]).view({-1, hidden_size_});
    o = mlp_0_(o);
    o = mlp_gelu_(o);
    o = mlp_2_(o);
    return {o};
  }
};

class VisionMlp final : public nn::Module {
  int32_t dim_;
  int32_t hidden_dim_;

  nn::QuickGELU act_;
  nn::Linear fc_1_;
  nn::Linear fc_2_;

 public:
  VisionMlp() = default;

  inline VisionMlp(const std::string& name, const Qwen2VLConfig& cfg) : nn::Module(name) {
    dim_ = cfg.visual_embed_dim;
    hidden_dim_ = cfg.visual_embed_dim * cfg.visual_mlp_ratio;

    fc_1_ = reg<nn::Linear>("fc1", dim_, hidden_dim_);
    fc_2_ = reg<nn::Linear>("fc2", hidden_dim_, dim_);
    act_ = reg<nn::QuickGELU>("act");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    return {fc_2_(act_(fc_1_(inputs[0])))};
  }
};

class VisionAttention final : public nn::Module {
  int32_t dim_;
  int32_t num_heads_;
  int32_t head_dim_;
  int32_t num_key_value_groups = 1;
  float scaling = 0.f;

  nn::Linear qkv_;
  nn::Linear proj_;
  nn::Softmax softmax_;
  nn::VisionRoPE vision_rope_q_;
  nn::VisionRoPE vision_rope_k_;

 public:
  VisionAttention() = default;

  inline VisionAttention(const std::string& name, const Qwen2VLConfig& cfg) : nn::Module(name) {
    dim_ = cfg.visual_embed_dim;
    num_heads_ = cfg.visual_num_heads;
    head_dim_ = dim_ / num_heads_;
    scaling = std::sqrt(head_dim_);

    qkv_ = reg<nn::Linear>("qkv", dim_, dim_ * 3, true, cfg.linear_impl_type);
    proj_ = reg<nn::Linear>("proj", dim_, dim_, true, cfg.linear_impl_type);
    softmax_ = reg<nn::Softmax>("softmax", -1);

    vision_rope_q_ = reg<nn::VisionRoPE>("vision_rope_q", aops::VisionRoPEOpOptionsType::kQwen2VL,
                                         aops::Qwen2VLRoPEOpOptions{
                                             .dims = head_dim_,
                                             .spatial_merge_size = cfg.visual_spatial_merge_size,
                                             .theta = 10000.0,
                                         });
    vision_rope_k_ = reg<nn::VisionRoPE>("vision_rope_k", aops::VisionRoPEOpOptionsType::kQwen2VL,
                                         aops::Qwen2VLRoPEOpOptions{
                                             .dims = head_dim_,
                                             .spatial_merge_size = cfg.visual_spatial_merge_size,
                                             .theta = 10000.0,
                                         });
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    // hidden_states shape is [seq_length, dim]
    auto hidden_states = inputs[0];
    auto& grid_thw = inputs[1];
    auto visual_embedding_sin = inputs[2];
    auto visual_embedding_cos = inputs[3];

    auto seq_length = hidden_states.shape()[0];

    auto [query_states, key_states, value_states] =
        nn::functional::split<3>(qkv_(hidden_states).view({seq_length, 3, num_heads_, -1}).permute({1, 0, 2, 3}), 1, 0);

    // Input to Vision ROPE must be BSHD format
    // grid_thw shape is [n, 3], n is always 1 in this case.
    auto [query_states_roped, sin, cos] = vision_rope_q_(query_states, grid_thw, visual_embedding_sin, visual_embedding_cos);
    auto [key_states_roped, _, _] = vision_rope_k_(key_states, grid_thw, visual_embedding_sin, visual_embedding_cos);

    visual_embedding_sin = sin;
    visual_embedding_cos = cos;

    // Reassigned.
    query_states = query_states_roped;
    key_states = key_states_roped;

    // [B, H, S, D]
    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    // attention weight
    // [B=1, H, S, S]
    auto attn = nn::functional::matmul(query_states, key_states, false, true) * (1.f / sqrtf(head_dim_));
    attn = softmax_(attn);

    // attn output
    // [B=1, H, S, S] @ [B=1, H, S, D] -> [B=1, H, S, D]
    auto attn_output = nn::functional::matmul(attn, value_states);

    // [B=1, H, S, D] -> [B=1, S, H, D] -> [S, H * D]
    attn_output = attn_output.transpose(1, 2).view({seq_length, -1});
    attn_output = proj_(attn_output);
    return {
        attn_output,
        visual_embedding_sin,
        visual_embedding_cos,
    };
  }
};

class Qwen2VLVisionBlock final : public nn::Module {
  int mlp_hidden_dim_;

  nn::LayerNorm norm1_;
  nn::LayerNorm norm2_;

  VisionAttention attn_;
  VisionMlp mlp_;

 public:
  Qwen2VLVisionBlock() = default;

  inline Qwen2VLVisionBlock(const std::string& name, const Qwen2VLConfig& cfg) : nn::Module(name) {
    mlp_hidden_dim_ = cfg.visual_mlp_ratio * cfg.visual_embed_dim;
    norm1_ = reg<nn::LayerNorm>("norm1", std::vector<int32_t>{cfg.visual_embed_dim}, true, true, 1e-6);
    norm2_ = reg<nn::LayerNorm>("norm2", std::vector<int32_t>{cfg.visual_embed_dim}, true, true, 1e-6);
    attn_ = reg<VisionAttention>("attn", cfg);
    mlp_ = reg<VisionMlp>("mlp", cfg);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto grid_thw = inputs[1];
    auto visual_embedding_sin = inputs[2];
    auto visual_embedding_cos = inputs[3];

    auto res = attn_(norm1_(hidden_states), grid_thw);
    auto& a = res[0];
    visual_embedding_sin = res[1];
    visual_embedding_cos = res[2];

    hidden_states = hidden_states + a;
    hidden_states = hidden_states + mlp_(norm2_(hidden_states))[0];
    return {hidden_states, grid_thw, visual_embedding_sin, visual_embedding_cos};
  }
};

class Qwen2VisionTransformerPretrainedModel final : public nn::Module {
  PatchEmbed patch_embed_;
  PatchMerger patch_merger_;
  nn::ModuleList<Qwen2VLVisionBlock> blocks_;

 public:
  Qwen2VisionTransformerPretrainedModel() = default;

  Qwen2VisionTransformerPretrainedModel(const std::string& name, const Qwen2VLConfig& cfg) : nn::Module(name) {
    patch_embed_ = reg<PatchEmbed>("patch_embed", cfg);
    patch_merger_ = reg<PatchMerger>("merger", cfg);
    blocks_ = reg<nn::ModuleList<Qwen2VLVisionBlock>>("blocks", cfg.visual_depth, cfg);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto grid_thw = inputs[1];
    auto embedding_sin = inputs[2];
    auto embedding_cos = inputs[3];

    hidden_states = patch_embed_(hidden_states)[0];

    for (auto& b : blocks_.list()) {
      auto o = b(hidden_states, grid_thw, embedding_sin, embedding_cos);
      hidden_states = o[0];
      grid_thw = o[1];
      embedding_sin = o[2];
      embedding_cos = o[3];
    }

    hidden_states = patch_merger_(hidden_states)[0];

    return {hidden_states};
  }
};

class Qwen2VLMLP final : public nn::Module {
  nn::Linear gate_proj_;
  nn::Linear up_proj_;
  nn::Linear down_proj_;
  nn::SiLU silu_;

 public:
  Qwen2VLMLP() = default;
  Qwen2VLMLP(const std::string& name, const Qwen2VLConfig& cfg) : nn::Module(name) {
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

class Qwen2VLAttention final : public nn::Module {
  nn::Linear q_proj_;
  nn::Linear k_proj_;
  nn::Linear v_proj_;
  nn::Linear o_proj_;
  nn::MultimodalRoPE q_rope_;
  nn::MultimodalRoPE k_rope_;
  nn::CausalMask mask_;
  nn::Softmax softmax_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;

 public:
  Qwen2VLAttention() = default;

  Qwen2VLAttention(const std::string& name, const Qwen2VLConfig& cfg) : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = hidden_size_ / num_attention_heads_;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;

    q_proj_ = reg<nn::Linear>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, true, cfg.linear_impl_type);
    k_proj_ = reg<nn::Linear>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, true, cfg.linear_impl_type);
    v_proj_ = reg<nn::Linear>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, true, cfg.linear_impl_type);
    o_proj_ = reg<nn::Linear>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, false, cfg.linear_impl_type);

    q_rope_ = reg<nn::MultimodalRoPE>(
        "q_rope", aops::Qwen2VLMultimodalRoPEOpOptions{.rope_theta = cfg.rope_theta,
                                                       .max_position_embeddings = cfg.max_position_embeddings,
                                                       .mrope_section = cfg.mrope_section});
    k_rope_ = reg<nn::MultimodalRoPE>(
        "k_rope", aops::Qwen2VLMultimodalRoPEOpOptions{.rope_theta = cfg.rope_theta,
                                                       .max_position_embeddings = cfg.max_position_embeddings,
                                                       .mrope_section = cfg.mrope_section});

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    auto pos_ids = inputs[1];
    auto llm_embedding_sin = inputs[2];
    auto llm_embedding_cos = inputs[3];
    auto past_kv_cache = args[0].get<nn::StaticCache*>();

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

    // [B, H, S, D]
    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    // [B, H, S, D]
    auto [query_states_rope, sin, cos] = q_rope_(query_states, pos_ids, llm_embedding_sin, llm_embedding_cos);
    auto [key_states_rope, _, _] = k_rope_(key_states, pos_ids, llm_embedding_sin, llm_embedding_cos);
    query_states = query_states_rope;
    key_states = key_states_rope;
    llm_embedding_sin = sin;
    llm_embedding_cos = cos;

    // [B, H, S, D]
    auto [k, v] = past_kv_cache->updateKVCache(layer_idx_, key_states, value_states);
    key_states = k;
    value_states = v;

    Tensor attn;
    if (key_states.dtype() == kFloat32) {
      // attention weight
      // [B, H, S, S]
      attn = nn::functional::matmul(query_states, key_states, false, true) * (1.f / sqrtf(head_dim_));
      attn = mask_(attn);
      attn = softmax_(attn);
    } else if (key_states.dtype() == kFloat16) {
      attn = nn::functional::matmul(query_states.to(kFloat32), key_states.to(kFloat32), false, true) * (1.f / sqrtf(head_dim_));
      attn = mask_(attn);
      attn = softmax_(attn);
      attn = attn.to(kFloat16);
    }

    // attn output
    // [B, H, S, S] @ [B, H, S, D] -> [B, H, S, D]
    auto output = nn::functional::matmul(attn, value_states);
    // [B, H, S, D] -> [B, S, H, D] -> [B, S, H * D]
    output = output.transpose(1, 2).view({B, S, num_attention_heads_ * head_dim_});
    output = o_proj_(output);
    return {output, llm_embedding_sin, llm_embedding_cos};
  }

  int layer_idx_;
};

class Qwen2VLDecoder final : public nn::Module {
 public:
  Qwen2VLAttention self_attn_;
  Qwen2VLMLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  Qwen2VLDecoder() = default;

  Qwen2VLDecoder(const std::string& name, const Qwen2VLConfig& cfg) : nn::Module(name) {
    self_attn_ = reg<Qwen2VLAttention>("self_attn", cfg);
    mlp_ = reg<Qwen2VLMLP>("mlp", cfg);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto pos_ids = inputs[1];
    auto llm_embedding_sin = inputs[2];
    auto llm_embedding_cos = inputs[3];
    auto& kv_cache = args[0];

    auto x = input_layer_norm_(inputs[0]);
    auto res = self_attn_(x, pos_ids, llm_embedding_sin, llm_embedding_cos, kv_cache);
    x = res[0];
    llm_embedding_sin = res[1];
    llm_embedding_cos = res[2];
    auto tmp = x + inputs[0];
    x = post_attention_layer_norm_(tmp);
    x = mlp_(x)[0];
    x = x + tmp;
    return {x, llm_embedding_sin, llm_embedding_cos};
  }
};

class Qwen2VLText final : public nn::Module {
  nn::ModuleList<Qwen2VLDecoder> decode_blocks_;
  nn::RMSNorm norm_;
  nn::Param lm_head_;
  bool tie_word_embeddings_;

 public:
  Qwen2VLText() = default;

  Qwen2VLText(const std::string& name, const Qwen2VLConfig& cfg) : nn::Module(name) {
    tie_word_embeddings_ = cfg.tie_word_embeddings;

    decode_blocks_ = reg<nn::ModuleList<Qwen2VLDecoder>>("layers", cfg.num_hidden_layers, cfg);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }

    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
    embedding_ = reg<nn::Embedding>("embed_tokens", cfg.vocab_size, cfg.hidden_size);
    if (cfg.tie_word_embeddings) { lm_head_ = reg<nn::Param>("lm_head", "model.embed_tokens.weight"); }
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& blocks = decode_blocks_.list();

    // X is already embedded
    auto x = inputs[0];
    auto pos_ids = inputs[1];
    auto llm_embedding_sin = inputs[2];
    auto llm_embedding_cos = inputs[3];
    auto& kv_cache = args[0];

    for (auto& block : blocks) {
      auto o = block(x, pos_ids, llm_embedding_sin, llm_embedding_cos, kv_cache);
      x = o[0];
      llm_embedding_sin = o[1];
      llm_embedding_cos = o[2];
    }
    x = norm_(x);

    // clip x to one seq length
    {
      auto S = x.shape()[1];
      x = x[{kAll, {S - 1}, kAll}];
    }

    if (tie_word_embeddings_) {
      auto lm_head = lm_head_.weight();
      // x is [B, S, D], lm_head is [V, D]
      x = nn::functional::matmul(x, lm_head, false, true);
    }

    return {x};
  }

  nn::Embedding embedding_;
};

class Qwen2VLForCausalLM {
 public:
  explicit Qwen2VLForCausalLM(const Qwen2VLConfig& cfg) : cfg(cfg), llm("model", cfg), visual("visual", cfg) {
    kv_cache_ = nn::StaticCache(cfg.max_cache_length, cfg.num_hidden_layers,
                                cfg.num_attention_heads,                    // q_heads
                                cfg.num_key_value_heads,                    // kv_heads
                                cfg.hidden_size / cfg.num_attention_heads,  // kv_dims
                                kFloat32,                                   // k_dtype
                                kFloat32,                                   // v_dtype
                                kCPU,                                       // device_type
                                false                                       // use_fa2
    );
  }

  inline Qwen2VLForCausalLMOutputPast operator()(Qwen2VLForCausalLMOutputPast& past) {
    // Calculate the text embeddings
    auto input_embeddings = llm.embedding_(past.sequence);

    if (!past.img.isNil()) {
      // process img
      print("ViT Processing: ...");
      print("Image shape is:", past.img.shape());
      auto visual_embedding_sin = Tensor::nil();
      auto visual_embedding_cos = Tensor::nil();
      auto start_time = std::chrono::high_resolution_clock::now();
      auto visual_embeddings = visual(past.img, past.grid_thw, visual_embedding_sin, visual_embedding_cos)[0];
      auto end_time = std::chrono::high_resolution_clock::now();
      auto all_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
      print("ViT Processing: done, time cost: {} seconds", all_time.count());

      // Insert visual embeddings into llm's embedding
      int32_t vision_pad_token_start = -1;
      {
        auto input_ids = past.sequence;
        auto S = input_ids.shape()[1];
        auto input_ids_ptr = input_ids.ptr<int64_t>();
        for (int s = 0; s < S; ++s) {
          if (input_ids_ptr[s] == cfg.vision_token_id) {
            vision_pad_token_start = s;
            break;
          }
        }
        MLLM_RT_ASSERT(vision_pad_token_start != -1);
      }
      // input_embedding is [B, S, D]
      auto D = input_embeddings.shape()[2];

      // FIXME: maybe visual_embeddings.shape()[0];
      auto visual_sequence = visual_embeddings.shape()[1];
      visual_embeddings.copy2(
          input_embeddings[{kAll, {vision_pad_token_start, vision_pad_token_start + visual_sequence}, kAll}]);
    }

    getPositionIds(past, cfg);

    auto llm_embedding_sin = Tensor::nil();
    auto llm_embedding_cos = Tensor::nil();
    auto sequence = llm(input_embeddings, past.position_ids, llm_embedding_sin, llm_embedding_cos, AnyValue(&kv_cache_))[0];

    return {
        .sequence = sequence,
        .img = Tensor::nil(),
        .grid_thw = past.grid_thw,
        .position_ids = past.position_ids,
    };
  }

  inline void getPositionIds(Qwen2VLForCausalLMOutputPast& past, const Qwen2VLConfig& cfg) {
    // Input is [B, S, D]
    if (!past.img.isNil()) {  // Prefill
      past.position_ids = getPositionIdsPrefill(past.sequence, past.grid_thw, cfg);
    } else {  // Decode
      auto last_pos = *past.position_ids.offsettedPtr<int64_t>({0, 0, past.position_ids.shape()[2] - 1});
      past.position_ids = Tensor::empty({3, 1, 1}, kFloat32, kCPU).alloc();
      *past.position_ids.offsettedPtr<int64_t>({0, 0, 0}) = last_pos + 1;
      *past.position_ids.offsettedPtr<int64_t>({1, 0, 0}) = last_pos + 1;
      *past.position_ids.offsettedPtr<int64_t>({2, 0, 0}) = last_pos + 1;
    }
  }

  inline Tensor getPositionIdsPrefill(Tensor& input_ids, Tensor& image_grid_thw, const Qwen2VLConfig& cfg) {
    // Input is [B, S]
    MLLM_RT_ASSERT_EQ(input_ids.shape().size(), 2);
    // image_grid_thw is [num_images, 3]
    MLLM_RT_ASSERT_EQ(image_grid_thw.shape().size(), 2);

    auto B = input_ids.shape()[0];
    MLLM_RT_ASSERT_EQ(B, 1);
    auto S = input_ids.shape()[1];

    Tensor position_ids = Tensor::empty({3, B, S}, kInt64, kCPU).alloc();

    // Process text and visual
    // 1. Find the place of the first image token
    // Only one image is supported.
    int32_t vision_pad_token_start = -1;
    {
      auto input_ids_ptr = input_ids.ptr<int64_t>();
      for (int s = 0; s < S; ++s) {
        if (input_ids_ptr[s] == cfg.vision_token_id) {
          vision_pad_token_start = s;
          break;
        }
      }
      MLLM_RT_ASSERT(vision_pad_token_start != -1);
    }

    // 2. Calculate grid dimensions
    int img_t, img_h, img_w;
    int inputs_t, inputs_h, inputs_w;
    {
      auto image_grid_thw_ptr = image_grid_thw.ptr<int32_t>();
      img_t = image_grid_thw_ptr[0];
      img_h = image_grid_thw_ptr[1];
      img_w = image_grid_thw_ptr[2];

      inputs_t = img_t;
      inputs_h = img_h / cfg.visual_spatial_merge_size;
      inputs_w = img_w / cfg.visual_spatial_merge_size;
    }

    // 3. We assume the inputs format is: T T T V V V T T T
    int64_t current_max_position_id = 0;
    // 3.1 Handle text (Sys token as usual).
    {
      int64_t start_idx = current_max_position_id;
      for (int d = 0; d < 3; ++d) {
        auto position_ids_ptr = position_ids.offsettedPtr<int64_t>({d, 0, 0});
        for (int64_t k = 0; k < vision_pad_token_start; ++k) { position_ids_ptr[k] = start_idx + k; }
      }
      current_max_position_id = vision_pad_token_start - 1;
    }
    // 3.2 Handle image
    {
      int _cnt = 0;
      int64_t vision_start_id = current_max_position_id + 1;
      for (int64_t ti = 0; ti < inputs_t; ++ti) {
        for (int64_t hi = 0; hi < inputs_h; ++hi) {
          for (int64_t wi = 0; wi < inputs_w; ++wi) {
            *position_ids.offsettedPtr<int64_t>({0, 0, vision_pad_token_start + _cnt}) = vision_start_id + ti;

            *position_ids.offsettedPtr<int64_t>({1, 0, vision_pad_token_start + _cnt}) = vision_start_id + hi;

            *position_ids.offsettedPtr<int64_t>({2, 0, vision_pad_token_start + _cnt}) = vision_start_id + wi;

            _cnt++;
          }
        }
      }
      auto dim_0_tail =
          *position_ids.offsettedPtr<int64_t>({0, 0, vision_pad_token_start + inputs_t * inputs_h * inputs_w - 1});
      auto dim_1_tail =
          *position_ids.offsettedPtr<int64_t>({1, 0, vision_pad_token_start + inputs_t * inputs_h * inputs_w - 1});
      auto dim_2_tail =
          *position_ids.offsettedPtr<int64_t>({2, 0, vision_pad_token_start + inputs_t * inputs_h * inputs_w - 1});
      current_max_position_id = std::max({dim_0_tail, dim_1_tail, dim_2_tail});
    }
    // 3.3 Handle Prompt
    {
      const int64_t vision_token_count = inputs_t * inputs_h * inputs_w;
      const int64_t trailing_text_start_seq = vision_pad_token_start + vision_token_count;
      const int64_t trailing_text_count = S - trailing_text_start_seq;

      if (trailing_text_count > 0) {
        int64_t start_id = current_max_position_id + 1;
        for (int d = 0; d < 3; ++d) {
          auto position_ids_ptr = position_ids.offsettedPtr<int64_t>({d, 0, 0});
          for (int64_t k = 0; k < trailing_text_count; ++k) {
            const int64_t seq_idx = trailing_text_start_seq + k;
            position_ids_ptr[seq_idx] = start_id + k;
          }
        }
      }
    }

    return position_ids;
  }

  const Qwen2VLConfig& cfg;
  Qwen2VLText llm;
  Qwen2VisionTransformerPretrainedModel visual;

 private:
  nn::StaticCache kv_cache_;
};

}  // namespace mllm::models::qwen2vl
