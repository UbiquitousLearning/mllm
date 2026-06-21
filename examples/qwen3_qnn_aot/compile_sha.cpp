// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Benefits:
// 1. Reduces QNN AOT compilation time
// 2. Improves HTP runtime performance
// 3. Enables better memory locality per head
//
// Usage:
//   ./compile_sha -m /path/to/model.mllm -c /path/to/config.json -aot_cfg /path/to/qnn_aot_cfg.json

#include <mllm/mllm.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>

#include "compile_common.hpp"
#include "modeling_qwen_qnn_aot_sha.hpp"

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model file path.");
  auto& model_cfg_path = Argparse::add<std::string>("-c|--config").help("Model config file path.");
  auto& qnn_aot_cfg_files = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");
  auto& qnn_env_path = Argparse::add<std::string>("-qnn_env|--qnn_env_path")
                           .def("/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/")
                           .help("QNN AOT Environment path.");
  auto& output_context_path = Argparse::add<std::string>("-o|--output_context_name").help("Output QNN context path.");

  Argparse::parse(argc, argv);

  constexpr int kContextLength = 1024;

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  if (!qnn_aot_cfg_files.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No input aot config file path provided");
    Argparse::printHelp();
    return -1;
  }
  if (!output_context_path.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No output context path provided");
    Argparse::printHelp();
    return -1;
  }

  auto model_cfg = mllm::models::qwen3::Qwen3Config(model_cfg_path.get());

  // Load original parameters
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);

  // ============================================================================
  // Key Step: Prepare SHA parameters by slicing MHA weights
  // ============================================================================
  // This is the critical step that transforms MHA weights into SHA weights.
  // For each Q/K/V projection, we slice the weight matrix into per-head pieces.
  //
  // Original:  q_proj.weight [num_heads * head_dim, hidden_size, 1, 1]
  // SHA:       q_proj.{h}.weight [head_dim, hidden_size, 1, 1] for each head h
  //
  mllm::print("Preparing SHA parameters (slicing MHA weights)...");
  mllm::models::qwen3::sha::prepareParametersForSHA(params, model_cfg);
  mllm::print("SHA parameters prepared.");

  // Create SHA model
  auto model = mllm::models::qwen3::sha::Qwen3ForCausalLM_SHA(model_cfg);

  qwen3_qnn_aot::addCausalMaskParams(params);
  model.load(params);

  // Create Qnn AOT Model
  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(qnn_env_path.get(),
                                               mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg_files.get()));

  auto trace_and_dump = [&](int seq_len, const std::string& mir_path) {
    auto trace_inputs = qwen3_qnn_aot::makeTraceInputs(seq_len, kContextLength, model_cfg, params);
    mllm::print("Tracing SHA model (seq=" + std::to_string(seq_len) + ")...");
    auto ir = model.trace(trace_inputs, {});
    mllm::print("SHA model traced successfully.");
    mllm::ir::PassManager pm(ir["model"]);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
    pm.run();
    mllm::redirect(mir_path, [&]() { mllm::print(ir["model"]); });
  };

  trace_and_dump(32, "qwen3_qnn_aot_sha_32.mir");
  trace_and_dump(1, "qwen3_qnn_aot_sha_1.mir");

  qnn_aot_env.saveContext("context.0", output_context_path.get());

  mllm::print("SHA compilation completed successfully!");
  mllm::print("Output files:");
  mllm::print("  - qwen3_qnn_aot_sha_32.mir (IR dump for seq=32)");
  mllm::print("  - qwen3_qnn_aot_sha_1.mir (IR dump for seq=1)");
  mllm::print("  - " + output_context_path.get() + " (QNN context)");
});
