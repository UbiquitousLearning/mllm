#include <mllm/mllm.hpp>
#include <mllm/models/qwen2vl/modeling_qwen2vl_traceable.hpp>
#include <mllm/models/qwen2vl/configuration_qwen2vl.hpp>
#include <mllm/models/qwen2vl/tokenization_qwen2vl.hpp>
#include <mllm/models/qwen2vl/image_preprocessor_qwen2vl.hpp>

#include <mllm/compile/ir/Trace.hpp>
#include <mllm/compile/PassManager.hpp>
#include <mllm/compile/passes/LLMCanonicalizationPipeline.hpp>
#include <mllm/compile/passes/ProgramLoweringPipeline.hpp>

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory").required(true);
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
    auto qwen2vl_cfg = mllm::models::qwen2vl::Qwen2VLConfig(config_path.get());
    auto qwen2vl_tokenizer = mllm::models::qwen2vl::Qwen2VLTokenizer(tokenizer_path.get());
    auto qwen2vl = mllm::models::qwen2vl::Qwen2VLForCausalLM(qwen2vl_cfg);

    auto param = mllm::load(model_path.get(), file_version);
    qwen2vl.llm.load(param);
    qwen2vl.visual.load(param);
    try {
      fmt::print("🔄 Processing...\n");
      auto inputs = qwen2vl_tokenizer.convertMessage({.prompt = "free", .img_file_path = "/Volumes/D/mllm/.tmp/gafei.jpeg"});
      fmt::print("\n🤖 Compiling: \n");

      auto irs = qwen2vl.trace(inputs, {});

      // Compile llm model
      {
        mllm::ir::PassManager pm(irs["model"]);
        pm.reg(mllm::ir::createLLMCanonicalizationPipeline({
            .auxiliary_dbg_info = false,
            .enable_eager_memory_solver = true,
        }));
        pm.reg(mllm::ir::createProgramLoweringPipeline());
        pm.run();
        mllm::print(irs["model"]);
      }

      // Compile visual model
      {
        mllm::ir::PassManager pm(irs["visual"]);
        pm.reg(mllm::ir::createLLMCanonicalizationPipeline({
            .auxiliary_dbg_info = false,
            .enable_eager_memory_solver = true,
        }));
        pm.reg(mllm::ir::createProgramLoweringPipeline());
        pm.run();
        mllm::print(irs["visual"]);
      }
    } catch (const std::exception& e) { fmt::print("\n❌ Error: {}\n{}\n", e.what(), std::string(60, '-')); }
  }

  mllm::print("\n");
  mllm::memoryReport();
})
