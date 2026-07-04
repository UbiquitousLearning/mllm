// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>

#include "mllm/mllm.hpp"
#include "mllm/nn/Nn.hpp"
#include "mllm/nn/Module.hpp"
#include "mllm/nn/Functional.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Enumerate.hpp"
#include "mllm/compile/ir/Trace.hpp"
#include "mllm/models/ARGeneration.hpp"
#include "mllm/models/qwen2vl/configuration_qwen2vl.hpp"

namespace mllm::models::qwen2vl::qnn_aot {

struct DebugOutputConfig {
  bool dump_block_outputs = false;
  bool dump_layer0_outputs = false;
  bool key_cache_uint16 = false;
};

namespace ptq {

inline Tensor QDQ_CONSTANT(nn::Module* m, Tensor in, const std::string& qdq_name_in_pytorch) {
  auto scale_name = qdq_name_in_pytorch + ".scale";
  auto zp_name = qdq_name_in_pytorch + ".zero_point";

  switch (in.dtype()) {
    case kFloat32:
    case kUInt16PerTensorAsy: {
      auto scale = m->getTopParameterFile()->pull(scale_name);
      auto zp = m->getTopParameterFile()->pull(zp_name);
      in.attach("scale", scale.impl(), true);
      in.attach("zero_point", zp.impl(), true);
      break;
    }
    default: {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Can't Process dtype={}", nameOfType(in.dtype()));
    }
  }
  return in;
}

inline Tensor QDQ(nn::Module* m, Tensor in, const std::string& qdq_name_in_pytorch) {
  auto scale_name = qdq_name_in_pytorch + ".fake_quant.scale";
  auto zp_name = qdq_name_in_pytorch + ".fake_quant.zero_point";
  if (!m->getModuleName().empty()) {
    scale_name = m->getModuleName() + "." + scale_name;
    zp_name = m->getModuleName() + "." + zp_name;
  }

  switch (in.dtype()) {
    case kUInt16PerTensorAsy:
    case kFloat32: {
      auto scale = m->getTopParameterFile()->pull(scale_name);
      auto zp = m->getTopParameterFile()->pull(zp_name);
      in.attach("scale", scale.impl(), true);
      in.attach("zero_point", zp.impl(), true);
      break;
    }
    default: {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Can't Process dtype={}", nameOfType(in.dtype()));
    }
  }
  return in;
}

inline Tensor QDQ_KV(nn::Module* m, Tensor in, const std::string& qdq_name_in_pytorch) {
  auto scale_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.scale";
  auto zp_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.zero_point";

  switch (in.dtype()) {
    case kUInt8PerTensorSym: {
      auto scale = m->getTopParameterFile()->pull(scale_name);
      auto zp = m->getTopParameterFile()->pull(zp_name);
      MLLM_RT_ASSERT_EQ(zp.item<mllm_int32_t>(), 128);
      auto new_zp = Tensor::constant(128, kInt32).setName(zp_name).setMemType(kParamsNormal);
      in.attach("scale", scale.impl(), true);
      in.attach("zero_point", new_zp.impl(), true);
      break;
    }
    default: {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Can't Process dtype={}", nameOfType(in.dtype()));
    }
  }
  return in;
}

inline Tensor QDQ_KV_UInt16Sym(nn::Module* m, Tensor in, const std::string& qdq_name_in_pytorch) {
  auto scale_name = m->getModuleName() + "." + qdq_name_in_pytorch + ".fake_quant.scale";

  switch (in.dtype()) {
    case kUInt16PerTensorSym: {
      auto scale = m->getTopParameterFile()->pull(scale_name);
      in.attach("scale", scale.impl(), true);
      break;
    }
    default: {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Can't Process dtype={}", nameOfType(in.dtype()));
    }
  }
  return in;
}

}  // namespace ptq

inline void appendAttentionHeadOutputs(std::vector<Tensor>& outputs,
                                       Tensor x,
                                       const std::string& base_name,
                                       int32_t num_heads) {
  for (int32_t h = 0; h < num_heads; ++h) {
    outputs.push_back(x.slice({kAll, {h, h + 1}, kAll, kAll}, true).setName(base_name + "_h" + std::to_string(h)));
  }
}

inline Tensor rotateHalf(Tensor x, nn::Module* m, const std::string& qdq_name_in_pytorch) {
  auto D = x.size(-1);
  auto x1 = x.slice({kAll, kAll, kAll, {kAll, D / 2}}, /*ssa=*/true);
  auto x2 = x.slice({kAll, kAll, kAll, {D / 2, kAll}}, /*ssa=*/true);
  return nn::functional::concat({ptq::QDQ(m, -x2, qdq_name_in_pytorch), x1}, -1);
}

using vi32 = std::vector<int32_t>;
#define CONV2D_PROPERTY_NO_BIAS vi32{1, 1}, vi32{1, 1}, vi32{0, 0}, vi32{1, 1}, false, aops::Conv2DOpImplType::kQNN_LPBQ_w4a16o16_G32

inline Tensor makeQuantizedBias(nn::Module* m, const std::string& bias_name, const std::string& qdq_name_in_pytorch,
                                int32_t channels) {
  auto prefix = m->getModuleName();
  auto full_bias_name = prefix.empty() ? bias_name : prefix + "." + bias_name;
  auto scale_name = prefix.empty() ? qdq_name_in_pytorch + ".fake_quant.scale"
                                   : prefix + "." + qdq_name_in_pytorch + ".fake_quant.scale";
  auto zp_name = prefix.empty() ? qdq_name_in_pytorch + ".fake_quant.zero_point"
                                : prefix + "." + qdq_name_in_pytorch + ".fake_quant.zero_point";

  auto params = m->getTopParameterFile();
  auto bias = params->pull(full_bias_name);
  auto scale = params->pull(scale_name);
  auto zero_point = params->pull(zp_name);

  MLLM_RT_ASSERT_EQ(bias.numel(), channels);
  MLLM_RT_ASSERT_EQ(scale.numel(), 1);
  MLLM_RT_ASSERT_EQ(zero_point.numel(), 1);

  const auto scale_value = scale.item<mllm_fp32_t>();
  const auto zero_point_value = zero_point.item<mllm_int32_t>();
  auto quant_bias = Tensor::empty({1, 1, 1, channels}, kUInt16).alloc();

  for (int32_t i = 0; i < channels; ++i) {
    const auto quantized =
        static_cast<int32_t>(std::lround(bias.ptr<mllm_fp32_t>()[i] / scale_value)) + zero_point_value;
    quant_bias.ptr<mllm_uint16_t>()[i] = static_cast<mllm_uint16_t>(std::clamp(quantized, 0, 65535));
  }

  quant_bias = quant_bias.__unsafeSetDType(kUInt16PerTensorAsy);
  quant_bias.setName(full_bias_name + ".q_uint16");
  quant_bias.setMemType(kParamsNormal);
  quant_bias.attach("scale", scale.impl(), true);
  quant_bias.attach("zero_point", zero_point.impl(), true);
  return quant_bias;
}

class Qwen2VLMLP final : public nn::Module {
  nn::Conv2D gate_proj_;
  nn::Conv2D up_proj_;
  nn::Conv2D down_proj_;
  nn::SiLU silu_;
  int hidden_size_;
  int intermediate_size_;
  bool dump_layer0_outputs_ = false;

 public:
  Qwen2VLMLP() = default;
  Qwen2VLMLP(const std::string& name, const Qwen2VLConfig& cfg, bool dump_layer0_outputs = false) : nn::Module(name) {
    gate_proj_ = reg<nn::Conv2D>("gate_proj", cfg.hidden_size, cfg.intermediate_size, CONV2D_PROPERTY_NO_BIAS);
    silu_ = reg<nn::SiLU>("act");
    up_proj_ = reg<nn::Conv2D>("up_proj", cfg.hidden_size, cfg.intermediate_size, CONV2D_PROPERTY_NO_BIAS);
    down_proj_ = reg<nn::Conv2D>("down_proj", cfg.intermediate_size, cfg.hidden_size, CONV2D_PROPERTY_NO_BIAS);
    hidden_size_ = cfg.hidden_size;
    intermediate_size_ = cfg.intermediate_size;
    dump_layer0_outputs_ = dump_layer0_outputs;
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto x = inputs[0];
    x = ptq::QDQ(this, x, "up_proj_input_qdq");
    x = x.view({1, 1, -1, hidden_size_}, true);

    auto up_result = ptq::QDQ(this, up_proj_(x), "up_proj_output_qdq").view({1, -1, intermediate_size_}, true);
    auto gate_result = ptq::QDQ(this, gate_proj_(x), "gate_proj_output_qdq").view({1, -1, intermediate_size_}, true);
    auto gate_proj_result = gate_result;

    gate_result = ptq::QDQ(this, gate_result * ptq::QDQ(this, nn::functional::sigmoid(gate_result), "sigmoid_output_qdq"),
                           "act_output_qdq");
    auto gate_act_result = gate_result;

    auto o = ptq::QDQ(this, gate_result * up_result, "down_proj_input_qdq");
    auto down_proj_input = o;
    o = o.view({1, 1, -1, intermediate_size_}, true);
    o = down_proj_(o).view({1, -1, hidden_size_}, true);

    if (!dump_layer0_outputs_) { return {o}; }
    return {o,
            up_result.setName("layer0_mlp_up_proj_out"),
            gate_proj_result.setName("layer0_mlp_gate_proj_out"),
            gate_act_result.setName("layer0_mlp_gate_act_out"),
            down_proj_input.setName("layer0_mlp_down_proj_input")};
  }
};

class Qwen2VLAttention final : public nn::Module {
  nn::Conv2D q_proj_;
  nn::Conv2D k_proj_;
  nn::Conv2D v_proj_;
  nn::Conv2D o_proj_;
  nn::CausalMask mask_;
  nn::Softmax softmax_;

  int hidden_size_;
  int head_dim_;
  int num_attention_heads_;
  int num_key_value_heads_;
  int num_key_value_groups_;
  float scale_;
  bool dump_layer0_outputs_ = false;
  bool key_cache_uint16_ = false;

 public:
  Qwen2VLAttention() = default;

  Qwen2VLAttention(const std::string& name,
                   const Qwen2VLConfig& cfg,
                   bool dump_layer0_outputs = false,
                   bool key_cache_uint16 = false)
      : nn::Module(name) {
    hidden_size_ = cfg.hidden_size;
    num_attention_heads_ = cfg.num_attention_heads;
    num_key_value_heads_ = cfg.num_key_value_heads;
    head_dim_ = cfg.hidden_size / cfg.num_attention_heads;
    num_key_value_groups_ = num_attention_heads_ / num_key_value_heads_;
    scale_ = 1.f / sqrtf(static_cast<float>(head_dim_));
    dump_layer0_outputs_ = dump_layer0_outputs;
    key_cache_uint16_ = key_cache_uint16;

    q_proj_ = reg<nn::Conv2D>("q_proj", hidden_size_, head_dim_ * num_attention_heads_, CONV2D_PROPERTY_NO_BIAS);
    k_proj_ = reg<nn::Conv2D>("k_proj", hidden_size_, head_dim_ * num_key_value_heads_, CONV2D_PROPERTY_NO_BIAS);
    v_proj_ = reg<nn::Conv2D>("v_proj", hidden_size_, head_dim_ * num_key_value_heads_, CONV2D_PROPERTY_NO_BIAS);
    o_proj_ = reg<nn::Conv2D>("o_proj", head_dim_ * num_attention_heads_, hidden_size_, CONV2D_PROPERTY_NO_BIAS);

    mask_ = reg<nn::CausalMask>("mask");
    softmax_ = reg<nn::Softmax>("softmax", -1);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto hidden_states = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto causal_mask = inputs[3];
    auto past_key = inputs[4];
    auto past_value = inputs[5];

    hidden_states = ptq::QDQ(this, hidden_states, "q_proj_input_qdq");
    hidden_states = hidden_states.view({1, 1, -1, hidden_size_}, true);

    auto query_states = ptq::QDQ(this, q_proj_(hidden_states), "q_proj_output_qdq");
    auto key_states = ptq::QDQ(this, k_proj_(hidden_states), "k_proj_output_qdq");
    auto value_states = ptq::QDQ(this, v_proj_(hidden_states), "v_cast_to_int16_qdq");

    query_states =
        ptq::QDQ(this,
                 query_states
                     + makeQuantizedBias(this, "q_proj.bias", "q_proj_output_qdq", head_dim_ * num_attention_heads_),
                 "q_proj_output_qdq");
    key_states =
        ptq::QDQ(this,
                 key_states
                     + makeQuantizedBias(this, "k_proj.bias", "k_proj_output_qdq", head_dim_ * num_key_value_heads_),
                 "k_proj_output_qdq");
    value_states =
        ptq::QDQ(this,
                 value_states
                     + makeQuantizedBias(this, "v_proj.bias", "v_cast_to_int16_qdq", head_dim_ * num_key_value_heads_),
                 "v_cast_to_int16_qdq");

    auto query_states_flat = query_states.view({1, -1, num_attention_heads_ * head_dim_}, true);
    auto key_states_flat = key_states.view({1, -1, num_key_value_heads_ * head_dim_}, true);

    auto value_states_flat = value_states.view({1, -1, num_key_value_heads_ * head_dim_}, true);

    query_states = query_states.view({1, -1, num_attention_heads_, head_dim_}, /*ssa=*/true).transpose(1, 2);
    key_states = key_states.view({1, -1, num_key_value_heads_, head_dim_}, /*ssa=*/true).transpose(1, 2);
    value_states = value_states_flat.view({1, -1, num_key_value_heads_, head_dim_}, /*ssa=*/true).transpose(1, 2);

    auto cos = llm_embedding_cos.unsqueeze(1, true);
    auto sin = llm_embedding_sin.unsqueeze(1, true);
    query_states =
        ptq::QDQ(this,
                 ptq::QDQ(this, query_states * cos, "q_rope_mul_0_output_qdq")
                     + ptq::QDQ(this, rotateHalf(query_states, this, "q_rope_neg_half_qdq") * sin, "q_rope_mul_1_output_qdq"),
                 "q_rope_add_0_output_qdq");
    key_states =
        ptq::QDQ(this,
                 ptq::QDQ(this, key_states * cos, "k_rope_mul_0_output_qdq")
                     + ptq::QDQ(this, rotateHalf(key_states, this, "k_rope_neg_half_qdq") * sin, "k_rope_mul_1_output_qdq"),
                 "k_rope_add_0_output_qdq");
    auto query_states_rope_flat = query_states.transpose(1, 2).view({1, -1, num_attention_heads_ * head_dim_}, true);
    auto key_states_rope_flat = key_states.transpose(1, 2).view({1, -1, num_key_value_heads_ * head_dim_}, true);

    if (key_cache_uint16_) {
      key_states = key_states.to(kFloat32);
      key_states = key_states.to(kUInt16PerTensorSym);
      key_states = ptq::QDQ_KV_UInt16Sym(this, key_states, "k_rope_add_0_output_qdq");
      key_states = key_states.transpose(2, 3);
    } else {
      key_states = key_states.to(kFloat32);
      key_states = key_states.to(kUInt8PerTensorSym);
      key_states = ptq::QDQ_KV(this, key_states, "k_cast_to_int8_qdq");
      key_states = key_states.transpose(2, 3);
    }

    value_states = value_states.to(kFloat32);
    value_states = value_states.to(kUInt8PerTensorSym);
    value_states = ptq::QDQ_KV(this, value_states, "v_cast_to_int8_qdq");
    auto value_cache_states_flat =
        value_states.transpose(1, 2).view({1, -1, num_key_value_heads_ * head_dim_}, true);

    auto kh = nn::functional::concat({past_key, key_states}, -1);
    auto vh = nn::functional::concat({past_value, value_states}, 2);

    kh = kh.repeat(num_key_value_groups_, 1);
    vh = vh.repeat(num_key_value_groups_, 1);

    auto attn = ptq::QDQ(this, nn::functional::matmul(query_states, kh), "qk_matmul_output_qdq");
    auto qk_matmul_out = attn;
    auto scale = ptq::QDQ(this, Tensor::constant(scale_, kFloat32), "scaling_qdq");
    attn = ptq::QDQ(this, attn.mulConstant(scale), "mul_0_output_qdq");
    auto qk_scaled_out = attn;

    auto attn_min = ptq::QDQ(this, attn.min(-1, true), "reduce_min_output_qdq");
    auto minus_value = ptq::QDQ(this, Tensor::constant(-20, kFloat32), "neg_20_qdq");
    auto attn_vv = ptq::QDQ(this, attn_min.addConstant(minus_value), "minus_0_output_qdq");
    auto zero_constant = ptq::QDQ_CONSTANT(this, Tensor::constant(0.f, kFloat32), "constant_zero");
    attn = nn::functional::where(causal_mask.equalConstant(zero_constant), attn, attn_vv);
    attn = ptq::QDQ(this, attn, "where_attn_qdq");
    auto qk_masked_out = attn;
    attn = ptq::QDQ(this, nn::functional::softmax(attn, -1), "softmax_output_qdq");
    auto softmax_out = attn;
    auto attn_value_states = ptq::QDQ(this, nn::functional::matmul(attn, vh), "attn_value_matmul_output_qdq");
    auto y = attn_value_states.transpose(1, 2).view({1, 1, -1, num_attention_heads_ * head_dim_}, /*ssa=*/true);
    auto attn_value_states_flat = y.view({1, -1, num_attention_heads_ * head_dim_}, true);
    y = o_proj_(y).view({1, -1, hidden_size_}, true);

    if (!dump_layer0_outputs_) { return {y, key_states, value_states}; }

    auto ret = std::vector<Tensor>{y,
                                   key_states,
                                   value_states,
                                   query_states_flat.setName("layer0_q_proj_out_flat"),
                                   key_states_flat.setName("layer0_k_proj_out_flat"),
                                   value_states_flat.setName("layer0_v_proj_out_flat"),
                                   value_cache_states_flat.setName("layer0_v_cache_out_flat"),
                                   query_states_rope_flat.setName("layer0_q_rope_out_flat"),
                                   key_states_rope_flat.setName("layer0_k_rope_out_flat")};
    appendAttentionHeadOutputs(ret, qk_matmul_out, "layer0_qk_matmul_out_flat", num_attention_heads_);
    appendAttentionHeadOutputs(ret, qk_scaled_out, "layer0_qk_scaled_out_flat", num_attention_heads_);
    appendAttentionHeadOutputs(ret, qk_masked_out, "layer0_qk_masked_out_flat", num_attention_heads_);
    appendAttentionHeadOutputs(ret, softmax_out, "layer0_softmax_out_flat", num_attention_heads_);
    ret.push_back(attn_value_states_flat.setName("layer0_attn_value_out_flat"));
    ret.push_back(y.setName("layer0_o_proj_out"));
    return ret;
  }

  int layer_idx_;
};

class Qwen2VLDecoder final : public nn::Module {
 public:
  int layer_idx_;
  bool dump_layer0_outputs_ = false;
  Qwen2VLAttention self_attn_;
  Qwen2VLMLP mlp_;
  nn::RMSNorm input_layer_norm_;
  nn::RMSNorm post_attention_layer_norm_;

  Qwen2VLDecoder() = default;

  Qwen2VLDecoder(const std::string& name, const Qwen2VLConfig& cfg, DebugOutputConfig debug_outputs, int layer_idx) : nn::Module(name) {
    layer_idx_ = layer_idx;
    dump_layer0_outputs_ = debug_outputs.dump_layer0_outputs && layer_idx == 0;
    self_attn_ = reg<Qwen2VLAttention>("self_attn", cfg, dump_layer0_outputs_, debug_outputs.key_cache_uint16);
    mlp_ = reg<Qwen2VLMLP>("mlp", cfg, dump_layer0_outputs_);
    input_layer_norm_ = reg<nn::RMSNorm>("input_layernorm", cfg.rms_norm_eps);
    post_attention_layer_norm_ = reg<nn::RMSNorm>("post_attention_layernorm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto causal_mask = inputs[3];
    auto past_key = inputs[4];
    auto past_value = inputs[5];

    auto hidden_states = inputs[0];
    if (layer_idx_ != 0) { hidden_states = ptq::QDQ(this, hidden_states, "input_layernorm_input_qdq"); }
    auto residual = hidden_states;
    hidden_states = input_layer_norm_(hidden_states);
    auto attn_out = self_attn_(hidden_states, llm_embedding_sin, llm_embedding_cos, causal_mask, past_key, past_value);
    auto layer0_input_layernorm_out = hidden_states;
    hidden_states = attn_out[0];
    hidden_states = ptq::QDQ(this, residual + ptq::QDQ(this, hidden_states, "add_0_lhs_input_qdq"), "add_0_output_qdq");
    residual = hidden_states;
    hidden_states = post_attention_layer_norm_(hidden_states);
    auto layer0_post_attention_layernorm_out = hidden_states;
    auto mlp_out = mlp_(hidden_states);
    hidden_states = mlp_out[0];
    auto layer0_mlp_down_proj_out = hidden_states;
    hidden_states = residual + ptq::QDQ(this, hidden_states, "add_1_lhs_input_qdq");
    if (!dump_layer0_outputs_) { return {hidden_states, attn_out[1], attn_out[2]}; }

    auto ret = std::vector<Tensor>{hidden_states, attn_out[1], attn_out[2],
                                   layer0_input_layernorm_out.setName("layer0_input_layernorm_out")};
    for (size_t i = 3; i < attn_out.size(); ++i) { ret.push_back(attn_out[i]); }
    ret.push_back(layer0_post_attention_layernorm_out.setName("layer0_post_attention_layernorm_out"));
    ret.push_back(mlp_out[1]);
    ret.push_back(mlp_out[2]);
    ret.push_back(mlp_out[3]);
    ret.push_back(mlp_out[4]);
    ret.push_back(layer0_mlp_down_proj_out.setName("layer0_mlp_down_proj_out"));
    ret.push_back(hidden_states.setName("layer0_block_out"));
    return ret;
  }
};

class Qwen2VLText final : public nn::Module {
  nn::ModuleListWithIdx<Qwen2VLDecoder> decode_blocks_;
  nn::RMSNorm norm_;
  int32_t num_hidden_layers_;
  int32_t hidden_size_;
  DebugOutputConfig debug_outputs_;

 public:
  Qwen2VLText() = default;

  Qwen2VLText(const std::string& name, const Qwen2VLConfig& cfg, DebugOutputConfig debug_outputs = {}) : nn::Module(name) {
    num_hidden_layers_ = cfg.num_hidden_layers;
    hidden_size_ = cfg.hidden_size;
    debug_outputs_ = debug_outputs;
    decode_blocks_ = reg<nn::ModuleListWithIdx<Qwen2VLDecoder>>("layers", cfg.num_hidden_layers, cfg, debug_outputs_);
    for (auto [idx, b] : enumerate(decode_blocks_.list())) { b.self_attn_.layer_idx_ = idx; }
    norm_ = reg<nn::RMSNorm>("norm", cfg.rms_norm_eps);
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args) override {
    auto& blocks = decode_blocks_.list();
    auto x = inputs[0];
    auto llm_embedding_sin = inputs[1];
    auto llm_embedding_cos = inputs[2];
    auto causal_mask = inputs[3];

    std::vector<Tensor> keys;
    std::vector<Tensor> values;
    std::vector<Tensor> block_outputs;
    std::vector<Tensor> layer0_outputs;
    for (auto [index, block] : enumerate(blocks)) {
      auto pk = inputs[4 + index];
      auto pv = inputs[4 + index + num_hidden_layers_];
      auto out = block(x, llm_embedding_sin, llm_embedding_cos, causal_mask, pk, pv);
      x = out[0];
      if (debug_outputs_.dump_block_outputs) { block_outputs.push_back(x.setName("block_out_" + std::to_string(index))); }
      if (debug_outputs_.dump_layer0_outputs && index == 0) {
        for (size_t i = 3; i < out.size(); ++i) { layer0_outputs.push_back(out[i]); }
      }
      keys.push_back(out[1]);
      values.push_back(out[2]);
    }

    x = norm_(ptq::QDQ(this, x, "norm_input_qdq"));
    x = x.view({1, 1, -1, hidden_size_}, true);

    auto ret = std::vector<Tensor>{x};
    for (const auto& item : keys) { ret.push_back(item); }
    for (const auto& item : values) { ret.push_back(item); }
    for (const auto& item : block_outputs) { ret.push_back(item); }
    for (const auto& item : layer0_outputs) { ret.push_back(item); }
    return ret;
  }
};

class Qwen2VLForCausalLM : public ARGeneration, public nn::Module {
 public:
  explicit Qwen2VLForCausalLM(const Qwen2VLConfig& cfg, DebugOutputConfig debug_outputs = {})
      : cfg_(cfg), debug_outputs_(debug_outputs) {
    eos_token_id_ = cfg.end_of_text_token_id;
    max_length_ = cfg.max_cache_length;
    tie_word_embeddings_ = cfg.tie_word_embeddings;

    llm_ = reg<Qwen2VLText>("model", cfg, debug_outputs_);
    if (cfg.tie_word_embeddings) {
      lm_head_ = reg<nn::Conv2D>("lm_head", cfg.hidden_size, cfg.vocab_size, CONV2D_PROPERTY_NO_BIAS);
    }
  }

  IROutput trace(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override {
    auto input_embeddings = input.at("input_embeddings");
    auto llm_embedding_sin = input.at("llm_embedding_sin");
    auto llm_embedding_cos = input.at("llm_embedding_cos");
    auto causal_mask = input.at("causal_mask");

    std::vector<Tensor> kv_caches;
    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
      auto past_key_name = "past_key_" + std::to_string(i);
      if (!input.count(past_key_name)) { throw std::runtime_error("Missing KV cache: " + past_key_name); }
      kv_caches.push_back(input.at(past_key_name));
    }
    for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
      auto past_value_name = "past_value_" + std::to_string(i);
      if (!input.count(past_value_name)) { throw std::runtime_error("Missing KV cache: " + past_value_name); }
      kv_caches.push_back(input.at(past_value_name));
    }

    ir::lowlevel::traceStart();
    std::vector<Tensor> llm_inputs = {input_embeddings, llm_embedding_sin, llm_embedding_cos, causal_mask};
    llm_inputs.insert(llm_inputs.end(), kv_caches.begin(), kv_caches.end());

    auto llm_out = llm_(llm_inputs);
    auto logits = llm_out[0];
    const auto base_output_count = 1 + 2 * cfg_.num_hidden_layers;
    size_t expected_output_count = base_output_count;
    if (debug_outputs_.dump_block_outputs) { expected_output_count += cfg_.num_hidden_layers; }
    if (debug_outputs_.dump_layer0_outputs) { expected_output_count += 16 + 4 * cfg_.num_attention_heads; }
    if (debug_outputs_.dump_block_outputs || debug_outputs_.dump_layer0_outputs) {
      MLLM_RT_ASSERT_EQ(llm_out.size(), expected_output_count);
    }

    logits = lm_head_(ptq::QDQ(this, logits, "lm_head_input_qdq"));
    logits = ptq::QDQ(this, logits, "lm_head_output_qdq");
    auto llm_ir = ir::lowlevel::traceStop();

    return {{"model", llm_ir}};
  }

  ARGenerationOutputPast forward(const ARGenerationOutputPast& input, const ARGenerationArgs& args) override { return {}; }

 private:
  const Qwen2VLConfig& cfg_;
  Qwen2VLText llm_;
  nn::Conv2D lm_head_;
  bool tie_word_embeddings_;
  DebugOutputConfig debug_outputs_;
};

#undef CONV2D_PROPERTY_NO_BIAS

}  // namespace mllm::models::qwen2vl::qnn_aot
