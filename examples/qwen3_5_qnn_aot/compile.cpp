// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Qwen3.5 QNN AOT Compiler — Phase 1
//
// Compiles the 6 full attention layers (indices 3,7,11,15,19,23) into QNN
// context binaries. Each layer produces 2 graphs: one for prefill (N=32) and
// one for decode (N=1). Total: 12 QNN graphs in one shared context.
//
// GDN layers, embedding, and lm_head are NOT compiled here — they run on CPU
// at runtime (see aot_run.cpp).

#include <unordered_map>
#include <mllm/mllm.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>

#include "modeling_qwen3_5_qnn_aot.hpp"

using mllm::Argparse;
using namespace mllm::models::qwen3_5;

// Full attention layer indices for Qwen3.5 0.8B (every 4th layer starting from 3)
static constexpr int FULL_ATTN_LAYERS[] = {3, 7, 11, 15, 19, 23};
static constexpr int NUM_FULL_ATTN_LAYERS = 6;

static void traceAndCompileLayer(
    Qwen3_5SingleLayerForQNN& model,
    const Qwen3_5Config& cfg,
    mllm::qnn::aot::QnnAOTEnv& qnn_aot_env,
    const std::string& qnn_aot_cfg_path,
    mllm::ParameterFile::ptr_t& params,
    int attn_idx,       // 0-5 (index into FULL_ATTN_LAYERS)
    int actual_layer,   // actual layer index (3,7,11,15,19,23)
    int N,              // sequence length
    int CL              // context length (max cache)
) {
  // hidden_states: [B=1, S=N, D=hidden_size]
  auto hidden_states = mllm::Tensor::zeros({1, N, cfg.hidden_size}, mllm::kFloat32);

  // position_ids: [N]
  auto position_ids = mllm::Tensor::zeros({N}, mllm::kInt32);

  // causal_mask: [B=1, 1, N, CL]
  auto causal_mask = mllm::Tensor::zeros({1, 1, N, CL}, mllm::kUInt16);
  {
    causal_mask = causal_mask.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
    causal_mask.attach("scale", params->pull("causal_mask.scale").impl(), true);
    causal_mask.attach("zero_point", params->pull("causal_mask.zero_point").impl(), true);
  }

  // past_key: [B=1, num_kv_heads, head_dim, CL-N]
  // past_value: [B=1, num_kv_heads, CL-N, head_dim]
  auto past_key = mllm::Tensor::empty({1, cfg.num_key_value_heads, cfg.head_dim, CL - N}, mllm::kUInt8PerTensorSym);
  auto past_value = mllm::Tensor::empty({1, cfg.num_key_value_heads, CL - N, cfg.head_dim}, mllm::kUInt8PerTensorSym);

  // Attach KV cache scale/zp from the actual layer's quantization params
  std::string layer_prefix = "model.language_model.layers." + std::to_string(actual_layer);
  past_key.attach("scale", params->pull(layer_prefix + ".self_attn.k_cast_to_int8_qdq.fake_quant.scale").impl(), true);
  past_key.attach("zero_point", params->pull(layer_prefix + ".self_attn.k_cast_to_int8_qdq.fake_quant.zero_point").impl(), true);
  past_value.attach("scale", params->pull(layer_prefix + ".self_attn.v_cast_to_int8_qdq.fake_quant.scale").impl(), true);
  past_value.attach("zero_point", params->pull(layer_prefix + ".self_attn.v_cast_to_int8_qdq.fake_quant.zero_point").impl(), true);

  std::unordered_map<std::string, mllm::Tensor> trace_inputs;
  trace_inputs["hidden_states"] = hidden_states;
  trace_inputs["position_ids"] = position_ids;
  trace_inputs["causal_mask"] = causal_mask;
  trace_inputs["past_key"] = past_key;
  trace_inputs["past_value"] = past_value;

  auto ir = model.trace(trace_inputs, {});

  // Run AOT lowering pipeline
  mllm::ir::PassManager pm(ir["model"]);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_path, params));
  pm.run();

  // Dump MIR for debugging
  std::string mir_name = "qwen3_5_attn" + std::to_string(attn_idx) + "_s" + std::to_string(N) + ".mir";
  mllm::redirect(mir_name, [&]() { mllm::print(ir["model"]); });

  mllm::print("  Compiled layer {} (attn_idx={}) with N={}\n", actual_layer, attn_idx, N);
}

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Pre-baked model file path (from convert_weights.py)");
  auto& model_cfg_path = Argparse::add<std::string>("-c|--config").help("Model config file path (config_mllm.json)");
  auto& qnn_aot_cfg_path = Argparse::add<std::string>("-aot_cfg|--aot_config").help("QNN AOT Config file path");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path")
                            .def("/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/")
                            .help("QNN AOT Environment path");
  auto& context_len = Argparse::add<int>("--context_len").def(1024).help("Context length");
  auto& prefill_len = Argparse::add<int>("--prefill_len").def(32).help("Prefill chunk size");

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  int N_prefill = prefill_len.get();
  int N_decode = 1;
  int CL = context_len.get();

  auto cfg = Qwen3_5Config(model_cfg_path.get());
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);

  // Add constant params for causal mask
  params->push("causal_mask.scale", mllm::Tensor::constant(0.001 / 65535.f, mllm::kFloat32));
  params->push("causal_mask.zero_point", mllm::Tensor::constant(65535, mllm::kInt32));
  params->push("constant_zero.scale", mllm::Tensor::constant(0.001 / 65535.f, mllm::kFloat32));
  params->push("constant_zero.zero_point", mllm::Tensor::constant(65535, mllm::kInt32));

  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(
      qnn_env_path.get(),
      mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg_path.get()));

  mllm::print("Qwen3.5 QNN AOT Compile: {} full attention layers\n", NUM_FULL_ATTN_LAYERS);
  mllm::print("  Context length: {}, Prefill: {}, Decode: {}\n", CL, N_prefill, N_decode);

  // Compile each full attention layer with both seq lengths
  for (int attn_idx = 0; attn_idx < NUM_FULL_ATTN_LAYERS; ++attn_idx) {
    int actual_layer = FULL_ATTN_LAYERS[attn_idx];
    mllm::print("\nCompiling attention layer {} (attn_idx={})...\n", actual_layer, attn_idx);

    // Create per-layer model (loads weights for this specific layer)
    auto model = Qwen3_5SingleLayerForQNN(cfg, actual_layer);
    model.load(params);

    // Prefill graph (N=32)
    mllm::print("  Tracing prefill (N={})...\n", N_prefill);
    traceAndCompileLayer(model, cfg, qnn_aot_env, qnn_aot_cfg_path.get(), params,
                         attn_idx, actual_layer, N_prefill, CL);

    // Decode graph (N=1)
    mllm::print("  Tracing decode (N={})...\n", N_decode);
    traceAndCompileLayer(model, cfg, qnn_aot_env, qnn_aot_cfg_path.get(), params,
                         attn_idx, actual_layer, N_decode, CL);
  }

  // Save all graphs into one context binary
  qnn_aot_env.saveContext("context.0", "qwen3_5-0.8B-hybrid.bin");
  mllm::print("\nSaved QNN context binary: qwen3_5-0.8B-hybrid.bin\n");
});
