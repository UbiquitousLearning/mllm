#pragma once

#include <string>
#include <unordered_map>

#include <mllm/mllm.hpp>
#include <mllm/models/qwen3/configuration_qwen3.hpp>

namespace qwen3_qnn_aot {

template <typename ParamsT>
inline void addCausalMaskParams(const ParamsT& params) {
  params->push("causal_mask.scale", mllm::Tensor::constant(0.001 / 65535.f, mllm::kFloat32));
  params->push("causal_mask.zero_point", mllm::Tensor::constant(65535, mllm::kInt32));
  params->push("constant_zero.scale", mllm::Tensor::constant(0.001 / 65535.f, mllm::kFloat32));
  params->push("constant_zero.zero_point", mllm::Tensor::constant(65535, mllm::kInt32));
}

template <typename ParamsT>
inline std::unordered_map<std::string, mllm::Tensor> makeTraceInputs(int seq_len,
                                                                     int context_len,
                                                                     const mllm::models::qwen3::Qwen3Config& model_cfg,
                                                                     const ParamsT& params) {
  auto sequence = mllm::Tensor::zeros({1, seq_len}, mllm::kInt32);
  auto causal_mask = mllm::Tensor::zeros({1, 1, seq_len, context_len}, mllm::kUInt16);
  causal_mask = causal_mask.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
  causal_mask.attach("scale", params->pull("causal_mask.scale").impl(), true);
  causal_mask.attach("zero_point", params->pull("causal_mask.zero_point").impl(), true);

  std::unordered_map<std::string, mllm::Tensor> trace_inputs;
  trace_inputs["sequence"] = sequence;
  trace_inputs["causal_mask"] = causal_mask;

  for (int i = 0; i < model_cfg.num_hidden_layers; ++i) {
    auto past_key_name = "past_key_" + std::to_string(i);
    auto past_value_name = "past_value_" + std::to_string(i);

    trace_inputs[past_key_name] = mllm::Tensor::empty({
        1,
        model_cfg.num_key_value_heads,
        model_cfg.head_dim,
        context_len - seq_len,
    }, mllm::kUInt8PerTensorSym);
    trace_inputs[past_value_name] = mllm::Tensor::empty({
        1,
        model_cfg.num_key_value_heads,
        context_len - seq_len,
        model_cfg.head_dim,
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

}  // namespace qwen3_qnn_aot
