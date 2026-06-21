#include <mllm/mllm.hpp>
#include <mllm/models/qwen2_5vl/modeling_qwen2_5vl_traceable.hpp>
#include <mllm/models/qwen2_5vl/configuration_qwen2_5vl.hpp>
#include <mllm/models/qwen2vl/tokenization_qwen2vl.hpp>
#include <mllm/models/qwen2vl/image_preprocessor_qwen2vl.hpp>

#include <mllm/compile/ir/Trace.hpp>

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);

  Argparse::parse(argc, argv);

  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV1;
  if (model_version.get() == "v1") {
    file_version = mllm::ModelFileVersion::kV1;
  } else if (model_version.get() == "v2") {
    file_version = mllm::ModelFileVersion::kV2;
  }

  if (help.isSet()) {
    Argparse::printHelp();
    mllm::shutdownContext();
    return 0;
  }

  {
    auto qwen2_5vl_cfg = mllm::models::qwen2_5vl::Qwen2_5VLConfig(config_path.get());
    auto qwen2_5vl = mllm::models::qwen2_5vl::Qwen2_5VLForCausalLM(qwen2_5vl_cfg);

    auto param = mllm::load(model_path.get(), file_version);
    qwen2_5vl.llm.load(param);
    qwen2_5vl.visual.load(param);
  }

  mllm::print("\n");
  mllm::memoryReport();
})
