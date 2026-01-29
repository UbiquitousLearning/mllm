// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <unordered_map>
#include <mllm/mllm.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>

#include "modeling_llama_qnn_aot.hpp"

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model file path.");
  auto& model_cfg_path = Argparse::add<std::string>("-c|--config").help("Model config file path.");
  auto& qnn_aot_cfg_files = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");

  Argparse::parse(argc, argv);

  int N = 32;
  int CL = 1024;

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  if (!qnn_aot_cfg_files.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No input aot config file path provided");
    Argparse::printHelp();
    return -1;
  }

  auto model_cfg = mllm::models::llama3::Llama3Config(model_cfg_path.get());
  auto model = mllm::models::llama3::LlamaForCausalLM(model_cfg);
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  // Add params for causal mask
  {
    params->push("causal_mask.scale", mllm::Tensor::constant(0.001 / 65536.f, mllm::kFloat32));
    params->push("causal_mask.zero_point", mllm::Tensor::constant(65536, mllm::kInt8));
    params->push("constant_zero.scale", mllm::Tensor::constant(0.001 / 65536.f, mllm::kFloat32));
    params->push("constant_zero.zero_point", mllm::Tensor::constant(65536, mllm::kInt8));
  }
  model.load(params);

  // Create Qnn AOT Model
  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv("/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/",
                                               mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg_files.get()));

  // Model length 32.

  {
    // Sequence: [B, N]
    // past_key_i: [B, H, D, CL-N] for each layer i
    // past_value_i: [B, H, CL-N, D] for each layer i
    // causal_mask: [B, 1, N, CL]
    auto sequence = mllm::Tensor::zeros({1, N}, mllm::kInt32);
    auto causal_mask = mllm::Tensor::zeros({1, 1, N, CL}, mllm::kUInt16);

    // NOTE: force set causal mask to UInt16Asy
    // NOTE: Attach scale and zero point to causal mask
    {
      causal_mask = causal_mask.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
      causal_mask.attach("scale", params->pull("causal_mask.scale").impl(), true);
      causal_mask.attach("zero_point", params->pull("causal_mask.zero_point").impl(), true);
    }

    // Create KV cache inputs for all layers
    std::unordered_map<std::string, mllm::Tensor> trace_inputs;
    trace_inputs["sequence"] = sequence;
    trace_inputs["causal_mask"] = causal_mask;
    for (int i = 0; i < model_cfg.num_hidden_layers; ++i) {
      auto past_key_name = "past_key_" + std::to_string(i);
      auto past_value_name = "past_value_" + std::to_string(i);

      // clang-format off
    trace_inputs[past_key_name] = mllm::Tensor::empty({
        1,
        model_cfg.num_key_value_heads,
        model_cfg.head_dim,
        CL - N,
    }, mllm::kUInt8PerTensorSym);
    trace_inputs[past_value_name] = mllm::Tensor::empty({1, model_cfg.num_key_value_heads, CL - N, model_cfg.head_dim}, mllm::kUInt8PerTensorSym);
    
    trace_inputs[past_key_name].attach("scale", params->pull("model.layers." + std::to_string(i) + ".self_attn.k_cast_to_int8_qdq.fake_quant.scale").impl(), true);
    trace_inputs[past_key_name].attach("zero_point", params->pull("model.layers." + std::to_string(i) + ".self_attn.k_cast_to_int8_qdq.fake_quant.zero_point").impl(), true);

    trace_inputs[past_value_name].attach("scale", params->pull("model.layers." + std::to_string(i) + ".self_attn.v_cast_to_int8_qdq.fake_quant.scale").impl(), true);
    trace_inputs[past_value_name].attach("zero_point", params->pull("model.layers." + std::to_string(i) + ".self_attn.v_cast_to_int8_qdq.fake_quant.zero_point").impl(), true);
      // clang-format on
    }

    auto ir = model.trace(trace_inputs, {});

    mllm::ir::PassManager pm(ir["model"]);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
    pm.run();

    mllm::redirect("llama_qnn_aot_32.mir", [&]() { mllm::print(ir["model"]); });
  }

  // Model length 1.
  {
    N = 1;

    // Sequence: [B, N]
    // past_key_i: [B, H, D, CL-N] for each layer i
    // past_value_i: [B, H, CL-N, D] for each layer i
    // causal_mask: [B, 1, N, CL]
    auto sequence = mllm::Tensor::zeros({1, N}, mllm::kInt32);
    auto causal_mask = mllm::Tensor::zeros({1, 1, N, CL}, mllm::kUInt16);

    // NOTE: force set causal mask to UInt16Asy
    // NOTE: Attach scale and zero point to causal mask
    {
      causal_mask = causal_mask.__unsafeSetDType(mllm::kUInt16PerTensorAsy);
      causal_mask.attach("scale", params->pull("causal_mask.scale").impl(), true);
      causal_mask.attach("zero_point", params->pull("causal_mask.zero_point").impl(), true);
    }

    // Create KV cache inputs for all layers
    std::unordered_map<std::string, mllm::Tensor> trace_inputs;
    trace_inputs["sequence"] = sequence;
    trace_inputs["causal_mask"] = causal_mask;
    for (int i = 0; i < model_cfg.num_hidden_layers; ++i) {
      auto past_key_name = "past_key_" + std::to_string(i);
      auto past_value_name = "past_value_" + std::to_string(i);

      // clang-format off
    trace_inputs[past_key_name] = mllm::Tensor::empty({
        1,
        model_cfg.num_key_value_heads,
        model_cfg.head_dim,
        CL - N,
    }, mllm::kUInt8PerTensorSym);
    trace_inputs[past_value_name] = mllm::Tensor::empty({1, model_cfg.num_key_value_heads, CL - N, model_cfg.head_dim}, mllm::kUInt8PerTensorSym);
    
    trace_inputs[past_key_name].attach("scale", params->pull("model.layers." + std::to_string(i) + ".self_attn.k_cast_to_int8_qdq.fake_quant.scale").impl(), true);
    trace_inputs[past_key_name].attach("zero_point", params->pull("model.layers." + std::to_string(i) + ".self_attn.k_cast_to_int8_qdq.fake_quant.zero_point").impl(), true);

    trace_inputs[past_value_name].attach("scale", params->pull("model.layers." + std::to_string(i) + ".self_attn.v_cast_to_int8_qdq.fake_quant.scale").impl(), true);
    trace_inputs[past_value_name].attach("zero_point", params->pull("model.layers." + std::to_string(i) + ".self_attn.v_cast_to_int8_qdq.fake_quant.zero_point").impl(), true);
      // clang-format on
    }

    auto ir = model.trace(trace_inputs, {});

    mllm::ir::PassManager pm(ir["model"]);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
    pm.run();

    mllm::redirect("llama_qnn_aot_1.mir", [&]() { mllm::print(ir["model"]); });
  }

  qnn_aot_env.saveContext("context.0", "llama-lpbq.bin");
});
