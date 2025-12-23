// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <unordered_map>
#include <mllm/mllm.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>

#include "modeling_qwen_qnn_aot.hpp"

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model file path.");
  auto& model_cfg_path = Argparse::add<std::string>("-c|--config").help("Model config file path.");
  auto& qnn_aot_cfg_files = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");

  Argparse::parse(argc, argv);

  constexpr int N = 32;
  constexpr int CL = 1024;

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  if (!qnn_aot_cfg_files.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No input aot config file path provided");
    Argparse::printHelp();
    return -1;
  }

  auto model_cfg = mllm::models::qwen3::Qwen3Config(model_cfg_path.get());
  auto model = mllm::models::qwen3::Qwen3ForCausalLM(model_cfg);
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);

  // Gen sin and cos
  {
    auto inv = mllm::models::qwen3::makeRoPEInvFreq(model_cfg.head_dim, model_cfg.rope_theta);
    auto position_ids = mllm::Tensor::empty({1, CL}, mllm::kInt64, mllm::kCPU).alloc();
    auto position_ids_ptr = position_ids.ptr<int64_t>();
    for (int b = 0; b < 1; ++b) {
      for (int s = 0; s < CL; ++s) { position_ids_ptr[b * CL + s] = s; }
    }
    auto [rope_sin, rope_cos] = mllm::models::qwen3::makeRotaryPosEmbedding(position_ids, inv, 1.f);
    params->push("rope_sin", rope_sin);
    params->push("rope_cos", rope_cos);
  }
  model.load(params);

  // Sequence: [B, N]
  // past_key_i: [B, H, D, CL-N] for each layer i
  // past_value_i: [B, H, CL-N, D] for each layer i
  // causal_mask: [B, 1, N, CL]
  auto sequence = mllm::Tensor::zeros({1, N});
  auto causal_mask = mllm::Tensor::zeros({1, 1, N, CL});

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
    }, mllm::kInt8PerTensorSym);
    trace_inputs[past_value_name] = mllm::Tensor::empty({1, model_cfg.num_key_value_heads, CL - N, model_cfg.head_dim}, mllm::kInt8PerTensorSym);
    // clang-format on
  }

  auto ir = model.trace(trace_inputs, {});

  // Create Qnn AOT Model
  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv("/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/",
                                               mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg_files.get()));
  auto c = qnn_aot_env.createContext("name", true);

  mllm::ir::PassManager pm(ir["model"]);
  pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get()));
  pm.run();

  mllm::redirect("qwen3_qnn_aot.mir", [&]() { mllm::print(ir["model"]); });
});
