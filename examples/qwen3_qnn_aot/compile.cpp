// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <mllm/mllm.hpp>
#include "modeling_qwen_qnn_aot.hpp"

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model file path.");
  auto& model_cfg_path = Argparse::add<std::string>("-c|--config").help("Model config file path.");
  auto& qnn_aot_cfg_files = Argparse::add<std::string>("-aot_cfg|--aot_config").help("AOT Config file path.");

  Argparse::parse(argc, argv);

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
  model.load(mllm::load(model_path.get(), mllm::ModelFileVersion::kV2));

  constexpr int N = 32;
  constexpr int CL = 1024;
  // Sequence: [B, N]
  // past_key: [B, H, D, CL-N]
  // past_value: [B, H, CL-N, D]
  // causal_mask: [B, 1, N, CL]
  auto sequence = mllm::Tensor::zeros({1, N});
  auto past_key = mllm::Tensor::zeros({
      1,
      model_cfg.num_key_value_heads,
      model_cfg.head_dim,
      CL - N,
  });
  auto past_value = mllm::Tensor::zeros({1, model_cfg.num_key_value_heads, CL - N, model_cfg.head_dim});
  auto causal_mask = mllm::Tensor::zeros({1, 1, N, CL});
  auto ir = model.trace(
      {
          {"sequence", sequence},
          {"past_key", past_key},
          {"past_value", past_value},
          {"causal_mask", causal_mask},
      },
      {});
  mllm::redirect("qwen3_qnn_aot.mir", [&]() { mllm::print(ir["model"]); });
});
