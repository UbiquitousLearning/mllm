// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cmath>
#include <string>
#include <unordered_map>

#include <mllm/mllm.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/compile/ir/Trace.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>
#include <mllm/models/qwen2vl/tokenization_qwen2vl.hpp>

#include "modeling_qwen2vl_qnn_aot.hpp"
#include "visual_aot_helpers.hpp"

using mllm::Argparse;

namespace {

constexpr float kDefaultInputEmbeddingScale = 0.002563515f;
constexpr int32_t kDefaultInputEmbeddingZeroPoint = 15604;
constexpr float kDefaultVisualPatchInputScale = 4.4f / 65535.f;
constexpr int32_t kDefaultVisualPatchInputZeroPoint = 32768;
constexpr float kDefaultVisualSinCosScale = 2.0f / 65535.f;
constexpr int32_t kDefaultVisualSinCosZeroPoint = 32768;
constexpr float kDefaultVisualAttentionMaskScale = 10000.0f / 65535.f;
constexpr int32_t kDefaultVisualAttentionMaskZeroPoint = 65535;
const std::string kVisualFinalOutputQDQ = "visual.merger.mlp.2_output_qdq";

enum class VisualIODType {
  kUInt16,
  kFloat32,
  kFloat16,
};

std::string firstHybridBodyInputQDQName() { return "visual.blocks.0.attn.qkv_input_qdq"; }

VisualIODType parseVisualIODType(const std::string& dtype) {
  if (dtype == "uint16") { return VisualIODType::kUInt16; }
  if (dtype == "fp32" || dtype == "float32") { return VisualIODType::kFloat32; }
  if (dtype == "fp16" || dtype == "float16") { return VisualIODType::kFloat16; }
  MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--visual_io_dtype must be uint16, fp32, or fp16.");
}

mllm::DataTypes visualFloatDType(VisualIODType dtype) {
  switch (dtype) {
    case VisualIODType::kFloat32: return mllm::kFloat32;
    case VisualIODType::kFloat16: return mllm::kFloat16;
    case VisualIODType::kUInt16:
    default: return mllm::kFloat32;
  }
}

bool isRawFloatVisualIO(VisualIODType dtype) { return dtype == VisualIODType::kFloat32 || dtype == VisualIODType::kFloat16; }

template <typename ParamsT>
void addCausalMaskParams(const ParamsT& params) {
  params->push("causal_mask.scale", mllm::Tensor::constant(0.001 / 65535.f, mllm::kFloat32));
  params->push("causal_mask.zero_point", mllm::Tensor::constant(65535, mllm::kInt32));
  params->push("constant_zero.scale", mllm::Tensor::constant(0.001 / 65535.f, mllm::kFloat32));
  params->push("constant_zero.zero_point", mllm::Tensor::constant(65535, mllm::kInt32));
}

template <typename ParamsT>
void replaceParam(const ParamsT& params, const std::string& name, const mllm::Tensor& tensor) {
  if (params->has(name)) { params->remove(name); }
  params->push(name, tensor);
}

template <typename ParamsT>
void overrideVisualFinalOutputQDQ(const ParamsT& params, float scale, int32_t zero_point) {
  replaceParam(params, kVisualFinalOutputQDQ + ".fake_quant.scale", mllm::Tensor::constant(scale, mllm::kFloat32));
  replaceParam(params, kVisualFinalOutputQDQ + ".fake_quant.zero_point", mllm::Tensor::constant(zero_point, mllm::kInt32));
  fmt::print("[Qwen2VL AOT Compile] override {}: scale={:.9f}, zero_point={}\n",
             kVisualFinalOutputQDQ,
             scale,
             zero_point);
}

template <typename ParamsT>
void scaleVisualActivationQDQ(const ParamsT& params, float multiplier) {
  if (multiplier <= 0.0f || std::abs(multiplier - 1.0f) < 1e-6f) { return; }

  int32_t patched = 0;
  std::vector<std::string> names;
  names.reserve(params->dict().size());
  for (auto it = params->begin(); it != params->end(); ++it) {
    names.push_back(it->first);
  }

  for (const auto& name : names) {
    if (name.rfind("visual.", 0) != 0) { continue; }
    const std::string scale_suffix = ".fake_quant.scale";
    if (name.size() < scale_suffix.size() || name.compare(name.size() - scale_suffix.size(), scale_suffix.size(), scale_suffix) != 0) {
      continue;
    }
    if (name.find("_input_qdq.") == std::string::npos && name.find("_output_qdq.") == std::string::npos) { continue; }

    mllm::Tensor scale_tensor = params->pull(name);
    const auto old_scale = scale_tensor.item<mllm::mllm_fp32_t>();
    replaceParam(params, name, mllm::Tensor::constant(old_scale * multiplier, mllm::kFloat32));
    ++patched;
  }
  fmt::print("[Qwen2VL AOT Compile] scaled {} visual activation QDQ scale tensor(s) by {:.6f}\n", patched, multiplier);
}

template <typename ParamsT>
mllm::Tensor makeUInt16AsymTensor(const std::vector<int32_t>& shape,
                                  const std::string& scale_name,
                                  const std::string& zp_name,
                                  const ParamsT& params) {
  auto tensor = mllm::Tensor::empty(shape, mllm::kUInt16);
  tensor = tensor.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
  tensor.attach("scale", params->pull(scale_name).impl(), true);
  tensor.attach("zero_point", params->pull(zp_name).impl(), true);
  return tensor;
}

template <typename ParamsT>
mllm::Tensor makeUInt16SymTensor(const std::vector<int32_t>& shape,
                                 const std::string& scale_name,
                                 const ParamsT& params) {
  auto tensor = mllm::Tensor::empty(shape, mllm::kUInt16);
  tensor = tensor.__unsafeSetDType(mllm::kUInt16PerTensorSym);
  tensor.attach("scale", params->pull(scale_name).impl(), true);
  return tensor;
}

inline mllm::Tensor makeUInt16AsymTensor(const std::vector<int32_t>& shape, float scale, int32_t zero_point) {
  auto tensor = mllm::Tensor::empty(shape, mllm::kUInt16);
  tensor = tensor.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
  tensor.attach("scale", mllm::Tensor::constant(scale, mllm::kFloat32).impl(), true);
  tensor.attach("zero_point", mllm::Tensor::constant(zero_point, mllm::kInt32).impl(), true);
  return tensor;
}

mllm::Tensor makeVisualTraceTensor(const std::vector<int32_t>& shape,
                                   VisualIODType visual_io_dtype,
                                   float scale,
                                   int32_t zero_point) {
  if (visual_io_dtype == VisualIODType::kUInt16) { return makeUInt16AsymTensor(shape, scale, zero_point); }
  return mllm::Tensor::empty(shape, visualFloatDType(visual_io_dtype), mllm::kCPU).alloc();
}

class RawPatchEmbedLinear final : public mllm::nn::Module {
  int32_t patch_dim_ = 0;
  int32_t embed_dim_ = 0;

  mllm::nn::Linear proj_;

 public:
  RawPatchEmbedLinear() = default;

  RawPatchEmbedLinear(const std::string& name, const mllm::models::qwen2vl::Qwen2VLConfig& cfg)
      : mllm::nn::Module(name) {
    patch_dim_ = cfg.visual_in_chans * cfg.visual_temporal_patch_size * cfg.visual_patch_size * cfg.visual_patch_size;
    embed_dim_ = cfg.visual_embed_dim;
    proj_ = reg<mllm::nn::Linear>("proj", patch_dim_, embed_dim_, false, mllm::aops::LinearImplTypes::kDefault);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto hidden_states = inputs[0];
    hidden_states = hidden_states.view({-1, patch_dim_}, true);
    hidden_states = proj_(hidden_states).view({-1, embed_dim_}, true);
    return {hidden_states};
  }
};

class RawVisionMlpPrimitiveQuickGELU final : public mllm::nn::Module {
  int32_t dim_ = 0;
  int32_t hidden_dim_ = 0;

  mllm::nn::Linear fc_1_;
  mllm::nn::Linear fc_2_;

 public:
  RawVisionMlpPrimitiveQuickGELU() = default;

  RawVisionMlpPrimitiveQuickGELU(const std::string& name, const mllm::models::qwen2vl::Qwen2VLConfig& cfg)
      : mllm::nn::Module(name) {
    dim_ = cfg.visual_embed_dim;
    hidden_dim_ = cfg.visual_embed_dim * cfg.visual_mlp_ratio;
    fc_1_ = reg<mllm::nn::Linear>("fc1", dim_, hidden_dim_, true, mllm::aops::LinearImplTypes::kDefault);
    fc_2_ = reg<mllm::nn::Linear>("fc2", hidden_dim_, dim_, true, mllm::aops::LinearImplTypes::kDefault);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto x = fc_1_(inputs[0]);
    x = x * mllm::nn::functional::sigmoid(x * 1.702f);
    return {fc_2_(x)};
  }
};

class RawVisionAttentionMaskedAOTRewrite final : public mllm::nn::Module {
  int32_t dim_ = 0;
  int32_t num_heads_ = 0;
  int32_t head_dim_ = 0;

  mllm::nn::Linear qkv_;
  mllm::nn::Linear proj_;
  mllm::nn::Softmax softmax_;

 public:
  RawVisionAttentionMaskedAOTRewrite() = default;

  RawVisionAttentionMaskedAOTRewrite(const std::string& name, const mllm::models::qwen2vl::Qwen2VLConfig& cfg)
      : mllm::nn::Module(name) {
    dim_ = cfg.visual_embed_dim;
    num_heads_ = cfg.visual_num_heads;
    head_dim_ = dim_ / num_heads_;
    qkv_ = reg<mllm::nn::Linear>("qkv", dim_, dim_ * 3, true, mllm::aops::LinearImplTypes::kDefault);
    proj_ = reg<mllm::nn::Linear>("proj", dim_, dim_, true, mllm::aops::LinearImplTypes::kDefault);
    softmax_ = reg<mllm::nn::Softmax>("softmax", -1);
  }

  mllm::Tensor applyVisionRoPEPrimitive(mllm::Tensor x, mllm::Tensor visual_embedding_sin, mllm::Tensor visual_embedding_cos) {
    const int32_t half_dim = head_dim_ / 2;
    auto x1 = x.slice({mllm::kAll, mllm::kAll, mllm::kAll, {mllm::kAll, half_dim}}, true);
    auto x2 = x.slice({mllm::kAll, mllm::kAll, mllm::kAll, {half_dim, mllm::kAll}}, true);
    auto sin = visual_embedding_sin;
    auto cos = visual_embedding_cos;
    if (sin.rank() == 2) {
      sin = sin.view({1, -1, 1, half_dim}, true);
      cos = cos.view({1, -1, 1, half_dim}, true);
    } else {
      MLLM_RT_ASSERT_EQ(sin.rank(), 4);
      MLLM_RT_ASSERT_EQ(cos.rank(), 4);
    }
    auto y1 = x1 * cos + (-(x2 * sin));
    auto y2 = x1 * sin + x2 * cos;
    return mllm::nn::functional::concat({y1, y2}, -1);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto hidden_states = inputs[0];
    auto visual_embedding_sin = inputs[1];
    auto visual_embedding_cos = inputs[2];
    auto attention_mask = inputs.size() > 3 ? inputs[3] : mllm::Tensor::nil();

    auto qkv_states = qkv_(hidden_states).view({-1, 3, num_heads_, head_dim_}, true);
    auto query_states = qkv_states.slice({mllm::kAll, {0, 1}, mllm::kAll, mllm::kAll}, true).transpose(0, 1);
    auto key_states = qkv_states.slice({mllm::kAll, {1, 2}, mllm::kAll, mllm::kAll}, true).transpose(0, 1);
    auto value_states = qkv_states.slice({mllm::kAll, {2, 3}, mllm::kAll, mllm::kAll}, true).transpose(0, 1);

    query_states = applyVisionRoPEPrimitive(query_states, visual_embedding_sin, visual_embedding_cos);
    key_states = applyVisionRoPEPrimitive(key_states, visual_embedding_sin, visual_embedding_cos);

    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    auto attn = mllm::nn::functional::matmul(query_states, key_states, false, true)
                * (1.f / std::sqrt(static_cast<float>(head_dim_)));
    if (!attention_mask.isNil()) { attn = attn + attention_mask; }
    attn = softmax_(attn);

    auto attn_output = mllm::nn::functional::matmul(attn, value_states);
    attn_output = attn_output.transpose(1, 2).view({-1, dim_}, true);
    return {proj_(attn_output)};
  }
};

class RawQwen2VLVisionBlockAOTRewrite final : public mllm::nn::Module {
  mllm::nn::LayerNorm norm1_;
  mllm::nn::LayerNorm norm2_;

  RawVisionAttentionMaskedAOTRewrite attn_;
  RawVisionMlpPrimitiveQuickGELU mlp_;

 public:
  RawQwen2VLVisionBlockAOTRewrite() = default;

  RawQwen2VLVisionBlockAOTRewrite(const std::string& name, const mllm::models::qwen2vl::Qwen2VLConfig& cfg)
      : mllm::nn::Module(name) {
    norm1_ = reg<mllm::nn::LayerNorm>("norm1", std::vector<int32_t>{cfg.visual_embed_dim}, true, true, 1e-6);
    norm2_ = reg<mllm::nn::LayerNorm>("norm2", std::vector<int32_t>{cfg.visual_embed_dim}, true, true, 1e-6);
    attn_ = reg<RawVisionAttentionMaskedAOTRewrite>("attn", cfg);
    mlp_ = reg<RawVisionMlpPrimitiveQuickGELU>("mlp", cfg);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto hidden_states = inputs[0];
    auto visual_embedding_sin = inputs[1];
    auto visual_embedding_cos = inputs[2];
    auto attention_mask = inputs.size() > 3 ? inputs[3] : mllm::Tensor::nil();

    auto norm1_out = norm1_(hidden_states);
    mllm::Tensor attn_out;
    if (attention_mask.isNil()) {
      attn_out = attn_(norm1_out, visual_embedding_sin, visual_embedding_cos)[0];
    } else {
      attn_out = attn_(norm1_out, visual_embedding_sin, visual_embedding_cos, attention_mask)[0];
    }
    hidden_states = hidden_states + attn_out;
    hidden_states = hidden_states + mlp_(norm2_(hidden_states))[0];
    return {hidden_states};
  }
};

class RawPatchMergerAOTRewrite final : public mllm::nn::Module {
  int32_t hidden_size_ = 0;
  int32_t spatial_merge_size_ = 0;
  int32_t context_dim_ = 0;

  mllm::nn::LayerNorm ln_q_;
  mllm::nn::Linear mlp_0_;
  mllm::nn::Linear mlp_2_;
  mllm::nn::GELU mlp_gelu_;

 public:
  RawPatchMergerAOTRewrite() = default;

  RawPatchMergerAOTRewrite(const std::string& name, const mllm::models::qwen2vl::Qwen2VLConfig& cfg)
      : mllm::nn::Module(name) {
    context_dim_ = cfg.visual_embed_dim;
    spatial_merge_size_ = cfg.visual_spatial_merge_size;
    hidden_size_ = context_dim_ * spatial_merge_size_ * spatial_merge_size_;

    ln_q_ = reg<mllm::nn::LayerNorm>("ln_q", std::vector<int32_t>{context_dim_}, true, true, 1e-6);
    mlp_0_ = reg<mllm::nn::Linear>("mlp.0", hidden_size_, hidden_size_, true, mllm::aops::LinearImplTypes::kDefault);
    mlp_gelu_ = reg<mllm::nn::GELU>("mlp.gelu");
    mlp_2_ = reg<mllm::nn::Linear>("mlp.2", hidden_size_, cfg.hidden_size, true, mllm::aops::LinearImplTypes::kDefault);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto o = ln_q_(inputs[0]).view({-1, hidden_size_}, true);
    o = mlp_0_(o);
    o = mlp_gelu_(o);
    o = mlp_2_(o);
    return {o};
  }
};

class RawQwen2VisionTransformerPretrainedModelAOTRewrite final : public mllm::nn::Module {
  RawPatchEmbedLinear patch_embed_;
  RawPatchMergerAOTRewrite patch_merger_;
  mllm::nn::ModuleList<RawQwen2VLVisionBlockAOTRewrite> blocks_;
  int32_t start_block_ = 0;
  int32_t active_blocks_ = -1;
  bool skip_merger_ = false;
  bool skip_patch_embed_ = false;

 public:
  RawQwen2VisionTransformerPretrainedModelAOTRewrite() = default;

  RawQwen2VisionTransformerPretrainedModelAOTRewrite(const std::string& name,
                                                     const mllm::models::qwen2vl::Qwen2VLConfig& cfg,
                                                     int32_t start_block = 0,
                                                     int32_t active_blocks = -1,
                                                     bool skip_merger = false,
                                                     bool skip_patch_embed = false)
      : mllm::nn::Module(name) {
    start_block_ = start_block;
    active_blocks_ = active_blocks;
    skip_merger_ = skip_merger;
    skip_patch_embed_ = skip_patch_embed;
    patch_embed_ = reg<RawPatchEmbedLinear>("patch_embed", cfg);
    patch_merger_ = reg<RawPatchMergerAOTRewrite>("merger", cfg);
    blocks_ = reg<mllm::nn::ModuleList<RawQwen2VLVisionBlockAOTRewrite>>("blocks", cfg.visual_depth, cfg);
  }

  std::vector<mllm::Tensor> forward(const std::vector<mllm::Tensor>& inputs,
                                    const std::vector<mllm::AnyValue>& /*args*/) override {
    auto hidden_states = inputs[0];
    auto embedding_sin = inputs[1];
    auto embedding_cos = inputs[2];
    auto attention_mask = inputs.size() > 3 ? inputs[3] : mllm::Tensor::nil();

    if (!skip_patch_embed_) { hidden_states = patch_embed_(hidden_states)[0]; }
    auto num_blocks = active_blocks_ < 0 ? static_cast<int32_t>(blocks_.list().size()) : active_blocks_;
    MLLM_RT_ASSERT(start_block_ >= 0);
    MLLM_RT_ASSERT(num_blocks >= 0);
    MLLM_RT_ASSERT(start_block_ + num_blocks <= static_cast<int32_t>(blocks_.list().size()));
    for (int32_t i = 0; i < num_blocks; ++i) {
      if (attention_mask.isNil()) {
        hidden_states = blocks_.list()[start_block_ + i](hidden_states, embedding_sin, embedding_cos)[0];
      } else {
        hidden_states = blocks_.list()[start_block_ + i](hidden_states, embedding_sin, embedding_cos, attention_mask)[0];
      }
    }
    if (!skip_merger_) { hidden_states = patch_merger_(hidden_states)[0]; }
    return {hidden_states};
  }
};

template <typename ParamsT>
std::unordered_map<std::string, mllm::Tensor> makeTraceInputs(int seq_len,
                                                             int context_len,
                                                             const mllm::models::qwen2vl::Qwen2VLConfig& model_cfg,
                                                             const ParamsT& params,
                                                             bool override_input_embedding_qp,
                                                             float input_embedding_scale,
                                                             int32_t input_embedding_zero_point,
                                                             bool key_cache_uint16) {
  const int head_dim = model_cfg.hidden_size / model_cfg.num_attention_heads;

  std::unordered_map<std::string, mllm::Tensor> trace_inputs;

  if (override_input_embedding_qp) {
    trace_inputs["input_embeddings"] =
        makeUInt16AsymTensor({1, seq_len, model_cfg.hidden_size}, input_embedding_scale, input_embedding_zero_point);
  } else {
    trace_inputs["input_embeddings"] = makeUInt16AsymTensor(
        {1, seq_len, model_cfg.hidden_size}, "model.embed_tokens.scale", "model.embed_tokens.zero_point", params);
  }
  trace_inputs["llm_embedding_sin"] =
      makeUInt16AsymTensor({1, seq_len, head_dim}, "model.sin_embedding_input_qdq.fake_quant.scale",
                           "model.sin_embedding_input_qdq.fake_quant.zero_point", params);
  trace_inputs["llm_embedding_cos"] =
      makeUInt16AsymTensor({1, seq_len, head_dim}, "model.cos_embedding_input_qdq.fake_quant.scale",
                           "model.cos_embedding_input_qdq.fake_quant.zero_point", params);
  trace_inputs["causal_mask"] =
      makeUInt16AsymTensor({1, 1, seq_len, context_len}, "causal_mask.scale", "causal_mask.zero_point", params);

  for (int i = 0; i < model_cfg.num_hidden_layers; ++i) {
    auto past_key_name = "past_key_" + std::to_string(i);
    auto past_value_name = "past_value_" + std::to_string(i);

    if (key_cache_uint16) {
      trace_inputs[past_key_name] = makeUInt16SymTensor(
          {
              1,
              model_cfg.num_key_value_heads,
              head_dim,
              context_len - seq_len,
          },
          "model.layers." + std::to_string(i) + ".self_attn.k_rope_add_0_output_qdq.fake_quant.scale",
          params);
    } else {
      trace_inputs[past_key_name] = mllm::Tensor::empty({
          1,
          model_cfg.num_key_value_heads,
          head_dim,
          context_len - seq_len,
      }, mllm::kUInt8PerTensorSym);
      trace_inputs[past_key_name].attach("scale",
                                         params->pull("model.layers." + std::to_string(i)
                                                      + ".self_attn.k_cast_to_int8_qdq.fake_quant.scale")
                                             .impl(),
                                         true);
      trace_inputs[past_key_name].attach("zero_point",
                                         params->pull("model.layers." + std::to_string(i)
                                                      + ".self_attn.k_cast_to_int8_qdq.fake_quant.zero_point")
                                             .impl(),
                                         true);
    }
    trace_inputs[past_value_name] = mllm::Tensor::empty({
        1,
        model_cfg.num_key_value_heads,
        context_len - seq_len,
        head_dim,
    }, mllm::kUInt8PerTensorSym);

    trace_inputs[past_value_name].attach("scale",
                                         params->pull("model.layers." + std::to_string(i)
                                                      + ".self_attn.v_cast_to_int8_qdq.fake_quant.scale")
                                             .impl(),
                                         true);
    trace_inputs[past_value_name].attach("zero_point",
                                         params->pull("model.layers." + std::to_string(i)
                                                      + ".self_attn.v_cast_to_int8_qdq.fake_quant.zero_point")
                                             .impl(),
                                         true);
  }

  return trace_inputs;
}

void compileVisualBundleGraphs(mllm::qnn::aot::QnnAOTEnv& qnn_aot_env,
                               const std::string& visual_aot_cfg_path,
                               const mllm::ParameterFile::ptr_t& visual_params,
                               const mllm::models::qwen2vl::Qwen2VLConfig& visual_cfg,
                               int32_t visual_patch_tokens,
                               int32_t patch_flat_dim,
                               const std::string& bundle_layout,
                               const std::string& visual_ir_prefix,
                               const std::string& graph_suffix,
                               VisualIODType visual_io_dtype) {
  if (isRawFloatVisualIO(visual_io_dtype) && bundle_layout != "single") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "--visual_io_dtype=fp16/fp32 is currently supported only with --visual_bundle_layout=single.");
  }
  const int32_t half_dim = visual_cfg.visual_embed_dim / visual_cfg.visual_num_heads / 2;
  auto visual_embedding_sin = makeVisualTraceTensor({1, visual_patch_tokens, 1, half_dim},
                                                    visual_io_dtype,
                                                    kDefaultVisualSinCosScale,
                                                    kDefaultVisualSinCosZeroPoint);
  auto visual_embedding_cos = makeVisualTraceTensor({1, visual_patch_tokens, 1, half_dim},
                                                    visual_io_dtype,
                                                    kDefaultVisualSinCosScale,
                                                    kDefaultVisualSinCosZeroPoint);
  auto visual_attention_mask = makeVisualTraceTensor({1, 1, 1, visual_patch_tokens},
                                                     visual_io_dtype,
                                                     kDefaultVisualAttentionMaskScale,
                                                     kDefaultVisualAttentionMaskZeroPoint);

  const auto segments = qwen2vl_qnn_aot::makeVisualBundleSegments(bundle_layout,
                                                                  visual_cfg.visual_depth,
                                                                  visual_patch_tokens,
                                                                  patch_flat_dim,
                                                                  visual_cfg,
                                                                  graph_suffix);

  for (const auto& segment : segments) {
    fmt::print("\n{:=^72}\n", fmt::format(" Compile {} ", segment.graph_name));
    const bool quantized_patch_embed =
        !segment.skip_patch_embed && visual_params->has("visual.patch_embed.proj.weight")
        && visual_params->pull("visual.patch_embed.proj.weight").dtype() == mllm::kInt8;
    const auto segment_input_qdq =
        qwen2vl_qnn_aot::visualSegmentInputQDQName(segment, visual_cfg.visual_depth);
    auto segment_img = [&]() {
      if (isRawFloatVisualIO(visual_io_dtype)) {
        return mllm::Tensor::empty(segment.input_shape, visualFloatDType(visual_io_dtype), mllm::kCPU).alloc();
      }
      return quantized_patch_embed
                 ? makeUInt16AsymTensor(segment.input_shape, kDefaultVisualPatchInputScale, kDefaultVisualPatchInputZeroPoint)
                 : (!segment_input_qdq.empty() && visual_params->has(segment_input_qdq + ".fake_quant.scale")
                        ? makeUInt16AsymTensor(segment.input_shape,
                                               segment_input_qdq + ".fake_quant.scale",
                                               segment_input_qdq + ".fake_quant.zero_point",
                                               visual_params)
                        : mllm::Tensor::empty(segment.input_shape, mllm::kFloat32, mllm::kCPU).alloc());
    }();

    mllm::Tensor visual_output;
    mllm::ir::lowlevel::traceStart();
    if (isRawFloatVisualIO(visual_io_dtype)) {
      auto visual = RawQwen2VisionTransformerPretrainedModelAOTRewrite("visual",
                                                                       visual_cfg,
                                                                       segment.start_block,
                                                                       segment.visual_blocks,
                                                                       segment.skip_merger,
                                                                       segment.skip_patch_embed);
      visual.load(visual_params);
      visual_output = segment.visual_blocks > 0
                          ? mllm::ir::lowlevel::traceModule(visual,
                                                            segment_img,
                                                            visual_embedding_sin,
                                                            visual_embedding_cos,
                                                            visual_attention_mask)[0]
                          : mllm::ir::lowlevel::traceModule(visual, segment_img, visual_embedding_sin, visual_embedding_cos)[0];
    } else {
      auto visual = qwen2vl_qnn_aot::Qwen2VisionTransformerPretrainedModelAOTRewrite("visual",
                                                                                    visual_cfg,
                                                                                    segment.start_block,
                                                                                    segment.visual_blocks,
                                                                                    segment.skip_merger,
                                                                                    segment.skip_patch_embed);
      visual.load(visual_params);
      visual_output = segment.visual_blocks > 0
                          ? mllm::ir::lowlevel::traceModule(visual,
                                                            segment_img,
                                                            visual_embedding_sin,
                                                            visual_embedding_cos,
                                                            visual_attention_mask)[0]
                          : mllm::ir::lowlevel::traceModule(visual, segment_img, visual_embedding_sin, visual_embedding_cos)[0];
    }
    const auto segment_output_qdq =
        qwen2vl_qnn_aot::visualGraphOutputQDQName(segment.graph_name, visual_cfg.visual_depth);
    if (!isRawFloatVisualIO(visual_io_dtype) && !segment_output_qdq.empty() && visual_params->has(segment_output_qdq + ".fake_quant.scale")
        && visual_params->has(segment_output_qdq + ".fake_quant.zero_point")) {
      visual_output.attach("scale", visual_params->pull(segment_output_qdq + ".fake_quant.scale").impl(), true);
      visual_output.attach("zero_point", visual_params->pull(segment_output_qdq + ".fake_quant.zero_point").impl(), true);
    }
    auto visual_ir = mllm::ir::lowlevel::traceStop();

    fmt::print("visual segment output shape: [{}, {}]\n", visual_output.shape()[0], visual_output.shape()[1]);
    const auto segment_ir_path = visual_ir_prefix + "." + segment.ir_name + ".mir";
    mllm::redirect(segment_ir_path, [&]() { mllm::print(visual_ir); });
    fmt::print("Visual IR dumped to: {}\n", segment_ir_path);

    mllm::ir::PassManager pm(visual_ir);
    pm.reg(mllm::qnn::aot::createQnnAOTSimpleLoweringPipeline(&qnn_aot_env,
                                                              visual_aot_cfg_path,
                                                              visual_params,
                                                              segment.graph_name));
    if (!pm.run()) {
      MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Visual QNN AOT lowering failed for graph {}.", segment.graph_name);
    }
  }
}

void compileVisualBundleGraphsFromImage(mllm::qnn::aot::QnnAOTEnv& qnn_aot_env,
                                        const std::string& visual_aot_cfg_path,
                                        const mllm::ParameterFile::ptr_t& visual_params,
                                        const mllm::models::qwen2vl::Qwen2VLConfig& visual_cfg,
                                        const std::string& tokenizer_path,
                                        const std::string& image_path,
                                        const std::string& prompt,
                                        const std::string& bundle_layout,
                                        const std::string& visual_ir_prefix,
                                        VisualIODType visual_io_dtype) {
  if (tokenizer_path.empty() || image_path.empty()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--include_visual_bundle requires --tokenizer and --image.");
  }

  auto tokenizer = mllm::models::qwen2vl::Qwen2VLTokenizer(tokenizer_path);
  auto inputs = tokenizer.convertMessage({.prompt = prompt, .img_file_path = image_path});
  auto img = inputs.at("img");
  compileVisualBundleGraphs(qnn_aot_env,
                            visual_aot_cfg_path,
                            visual_params,
                            visual_cfg,
                            img.shape()[0],
                            img.shape()[1],
                            bundle_layout,
                            visual_ir_prefix,
                            "",
                            visual_io_dtype);
}

}  // namespace

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model file path.");
  auto& model_cfg_path = Argparse::add<std::string>("-c|--config").help("Model config file path.");
  auto& qnn_aot_cfg_files = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path")
                           .def("/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/")
                           .help("QNN AOT Environment path.");
  auto& output_context_path = Argparse::add<std::string>("-o|--output_context_name").help("Output QNN context path.");
  auto& context_len = Argparse::add<int>("--context_len").help("QNN context length.").def(1024);
  auto& prefill_len = Argparse::add<int>("--prefill_len").help("Prefill graph sequence length.").def(32);
  auto& input_embedding_scale =
      Argparse::add<float>("--input_embedding_scale")
          .help("input_embeddings UInt16 scale. Defaults to the Qwen2-VL visual-safe wide QP; set both input embedding "
                "QP arguments to -1 to use model.embed_tokens QP.")
          .def(kDefaultInputEmbeddingScale);
  auto& input_embedding_zero_point =
      Argparse::add<int>("--input_embedding_zero_point")
          .help("input_embeddings UInt16 zero point. Defaults to the Qwen2-VL visual-safe wide QP; set both input "
                "embedding QP arguments to -1 to use model.embed_tokens QP.")
          .def(kDefaultInputEmbeddingZeroPoint);
  auto& dump_block_outputs =
      Argparse::add<bool>("--dump_block_outputs").help("Expose per-layer block_out tensors as graph outputs for debugging.");
  auto& dump_layer0_outputs =
      Argparse::add<bool>("--dump_layer0_outputs").help("Expose layer0 fine-grained tensors as graph outputs for debugging.");
  auto& key_cache_dtype =
      Argparse::add<std::string>("--key_cache_dtype").help("Key cache dtype for experimental contexts: uint8 or uint16.").def("uint8");
  auto& include_visual_bundle =
      Argparse::add<bool>("--include_visual_bundle")
          .help("Also compile the Qwen2-VL visual tower bundle graphs into the same QNN context.");
  auto& skip_llm_graphs =
      Argparse::add<bool>("--skip_llm_graphs")
          .help("Only compile requested auxiliary graphs, such as the visual bundle, and skip LLM prefill/decode graphs.");
  auto& visual_model_path = Argparse::add<std::string>("--visual_model")
                                .help("Optional FP32/W32A32 visual-capable .mllm for visual bundle graphs. Defaults to --model_path.");
  auto& visual_config_path = Argparse::add<std::string>("--visual_config")
                                 .help("Optional visual config. Defaults to --config.");
  auto& visual_aot_config_path = Argparse::add<std::string>("--visual_aot_config")
                                     .help("AOT config for visual bundle graphs. Defaults to --aot_config.");
  auto& visual_bundle_layout =
      Argparse::add<std::string>("--visual_bundle_layout")
          .def("6x8")
          .help("Visual bundle layout: single, 6x8, tail4, early2 or block1.");
  auto& visual_io_dtype =
      Argparse::add<std::string>("--visual_io_dtype")
          .def("uint16")
          .help("Visual graph input/output dtype: uint16 for quantized LPBQ visual, fp32/fp16 for raw float single visual graph.");
  auto& visual_bucket_grids = Argparse::add<std::string>("--visual_bucket_grids")
                                  .def("")
                                  .help("Comma-separated visual patch grid buckets HxW. Example: 10x16,12x16,26x36.");
  auto& visual_output_scale =
      Argparse::add<float>("--visual_output_scale")
          .def(-1.0f)
          .help("Override visual final output UInt16 scale for visual bundle graph compilation.");
  auto& visual_output_zero_point =
      Argparse::add<int>("--visual_output_zero_point")
          .def(-1)
          .help("Override visual final output UInt16 zero point for visual bundle graph compilation.");
  auto& visual_qdq_scale_multiplier =
      Argparse::add<float>("--visual_qdq_scale_multiplier")
          .def(1.0f)
          .help("Multiply visual *_input_qdq/*_output_qdq scale tensors before compiling visual graphs.");
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer").help("tokenizer.json path for visual input shape.");
  auto& image_path = Argparse::add<std::string>("-i|--image").help("image path for visual input shape.");
  auto& prompt = Argparse::add<std::string>("-p|--prompt").help("prompt text used with --image.").def("describe this picture");
  auto& visual_ir_prefix =
      Argparse::add<std::string>("--visual_ir_prefix").def("qwen2vl_visual_combined").help("Prefix for visual IR dumps.");

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }
  if (!model_path.isSet() || !model_cfg_path.isSet() || !qnn_aot_cfg_files.isSet() || !output_context_path.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Missing required argument.");
    Argparse::printHelp();
    return -1;
  }
  if (prefill_len.get() <= 0 || context_len.get() <= prefill_len.get()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Invalid context_len/prefill_len: {} / {}", context_len.get(),
                    prefill_len.get());
    return -1;
  }
  const bool override_input_embedding_qp = input_embedding_scale.get() > 0.0f || input_embedding_zero_point.get() >= 0;
  if (override_input_embedding_qp && (input_embedding_scale.get() <= 0.0f || input_embedding_zero_point.get() < 0)) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "input embedding override requires both --input_embedding_scale and --input_embedding_zero_point; "
                    "set both to -1 to use model.embed_tokens QP.");
  }
  const bool key_cache_uint16 = key_cache_dtype.get() == "uint16";
  if (!key_cache_uint16 && key_cache_dtype.get() != "uint8") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--key_cache_dtype must be uint8 or uint16.");
  }
  const bool override_visual_output_qp = visual_output_scale.get() > 0.0f || visual_output_zero_point.get() >= 0;
  if (override_visual_output_qp && (visual_output_scale.get() <= 0.0f || visual_output_zero_point.get() < 0)) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "visual output override requires both --visual_output_scale and --visual_output_zero_point.");
  }
  const auto parsed_visual_io_dtype = parseVisualIODType(visual_io_dtype.get());
  if (isRawFloatVisualIO(parsed_visual_io_dtype) && visual_bundle_layout.get() != "single") {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError,
                    "--visual_io_dtype=fp16/fp32 requires --visual_bundle_layout=single for now.");
  }

  auto model_cfg = mllm::models::qwen2vl::Qwen2VLConfig(model_cfg_path.get());
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  addCausalMaskParams(params);

  mllm::models::qwen2vl::qnn_aot::DebugOutputConfig debug_outputs{
      .dump_block_outputs = dump_block_outputs.isSet(),
      .dump_layer0_outputs = dump_layer0_outputs.isSet(),
      .key_cache_uint16 = key_cache_uint16,
  };
  auto model = mllm::models::qwen2vl::qnn_aot::Qwen2VLForCausalLM(model_cfg, debug_outputs);
  model.load(params);

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(qnn_env_path.get(),
                                               mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg_files.get()));

  auto trace_and_dump = [&](int seq_len, const std::string& mir_path) {
    mllm::print("Tracing Qwen2-VL LLM QNN AOT graph, seq=" + std::to_string(seq_len));
    auto trace_inputs =
        makeTraceInputs(seq_len, context_len.get(), model_cfg, params, override_input_embedding_qp,
                        input_embedding_scale.get(), input_embedding_zero_point.get(), key_cache_uint16);
    auto ir = model.trace(trace_inputs, {});
    mllm::print("Trace completed, lowering to QNN AOT.");

    mllm::ir::PassManager pm(ir["model"]);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
    pm.run();

    mllm::redirect(mir_path, [&]() { mllm::print(ir["model"]); });
    mllm::print("IR dumped to " + mir_path);
  };

  if (skip_llm_graphs.isSet() && !include_visual_bundle.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "--skip_llm_graphs requires --include_visual_bundle or another graph family.");
  }

  if (!skip_llm_graphs.isSet()) {
    trace_and_dump(prefill_len.get(), "qwen2vl_qnn_aot_" + std::to_string(prefill_len.get()) + ".mir");
    trace_and_dump(1, "qwen2vl_qnn_aot_1.mir");
  } else {
    mllm::print("Skipping LLM prefill/decode graph compilation.");
  }

  if (include_visual_bundle.isSet()) {
    const auto visual_model = visual_model_path.isSet() ? visual_model_path.get() : model_path.get();
    const auto visual_config = visual_config_path.isSet() ? visual_config_path.get() : model_cfg_path.get();
    const auto visual_aot_config = visual_aot_config_path.isSet() ? visual_aot_config_path.get() : qnn_aot_cfg_files.get();
    auto visual_cfg = mllm::models::qwen2vl::Qwen2VLConfig(visual_config);
    auto visual_params = mllm::load(visual_model, mllm::ModelFileVersion::kV2);
    if (override_visual_output_qp) {
      overrideVisualFinalOutputQDQ(visual_params, visual_output_scale.get(), visual_output_zero_point.get());
    }
    scaleVisualActivationQDQ(visual_params, visual_qdq_scale_multiplier.get());
    qwen2vl_qnn_aot::reshapePatchEmbedConv3DWeightForLinear(visual_params, visual_cfg);

    const int32_t patch_flat_dim =
        visual_cfg.visual_in_chans * visual_cfg.visual_temporal_patch_size * visual_cfg.visual_patch_size * visual_cfg.visual_patch_size;
    const auto buckets = qwen2vl_qnn_aot::parseVisualBucketGrids(visual_bucket_grids.get());
    if (!buckets.empty()) {
      const auto bucket_tokens = qwen2vl_qnn_aot::uniqueVisualBucketPatchTokens(buckets);
      fmt::print("Compiling {} visual bucket shape(s): ", bucket_tokens.size());
      for (size_t i = 0; i < bucket_tokens.size(); ++i) { fmt::print("{}{}", i == 0 ? "" : ",", bucket_tokens[i]); }
      fmt::print("\n");
      for (const int32_t patch_tokens : bucket_tokens) {
        compileVisualBundleGraphs(qnn_aot_env,
                                  visual_aot_config,
                                  visual_params,
                                  visual_cfg,
                                  patch_tokens,
                                  patch_flat_dim,
                                  visual_bundle_layout.get(),
                                  visual_ir_prefix.get(),
                                  qwen2vl_qnn_aot::visualGraphSuffixForPatchTokens(patch_tokens),
                                  parsed_visual_io_dtype);
      }
    } else {
      compileVisualBundleGraphsFromImage(qnn_aot_env,
                                         visual_aot_config,
                                         visual_params,
                                         visual_cfg,
                                         tokenizer_path.get(),
                                         image_path.get(),
                                         prompt.get(),
                                         visual_bundle_layout.get(),
                                         visual_ir_prefix.get(),
                                         parsed_visual_io_dtype);
    }
  }

  qnn_aot_env.saveContext("context.0", output_context_path.get());
  mllm::print("Qwen2-VL QNN AOT compilation completed.");
  mllm::print("Context: " + output_context_path.get());
});
