// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <mllm/mllm.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/backends/qnn/aot/QnnWrappersAPI.hpp>
#include <mllm/backends/qnn/aot/passes/AOTPipeline.hpp>
#include <mllm/backends/qnn/aot/QnnTargetMachineParser.hpp>

#include "compile_common.hpp"
#include "modeling_qwen_qnn_aot.hpp"

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
  auto model = mllm::models::qwen3::Qwen3ForCausalLM(model_cfg);
  auto params = mllm::load(model_path.get(), mllm::ModelFileVersion::kV2);
  qwen3_qnn_aot::addCausalMaskParams(params);
  model.load(params);

  // Create Qnn AOT Model
  auto qnn_aot_env = mllm::qnn::aot::QnnAOTEnv(qnn_env_path.get(),
                                               mllm::qnn::aot::parseQcomTargetMachineFromJSONFile(qnn_aot_cfg_files.get()));

  auto trace_and_dump = [&](int seq_len, const std::string& mir_path) {
    auto trace_inputs = qwen3_qnn_aot::makeTraceInputs(seq_len, kContextLength, model_cfg, params);
    auto ir = model.trace(trace_inputs, {});
    mllm::ir::PassManager pm(ir["model"]);
    pm.reg(mllm::qnn::aot::createQnnAOTLoweringPipeline(&qnn_aot_env, qnn_aot_cfg_files.get(), params));
    pm.run();
    mllm::redirect(mir_path, [&]() { mllm::print(ir["model"]); });
  };

  trace_and_dump(32, "qwen3_qnn_aot_32.mir");
  trace_and_dump(1, "qwen3_qnn_aot_1.mir");

  qnn_aot_env.saveContext("context.0", output_context_path.get());
});
