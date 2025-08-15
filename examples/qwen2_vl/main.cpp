#include <mllm/mllm.hpp>
#include <mllm/models/qwen2vl/modeling_qwen2vl.hpp>
#include <mllm/models/qwen2vl/tokenization_qwen2vl.hpp>
#include <mllm/models/qwen2vl/image_preprocessor_qwen2vl.hpp>

using mllm::Argparse;

int main(int argc, char** argv) {
  mllm::initializeContext();

  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_dir_path = Argparse::add<std::string>("-m|--model_path").help("Model directory").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    mllm::shutdownContext();
    return 0;
  }

  {
    auto qwen2vl_cfg = mllm::models::qwen2vl::Qwen2VLConfig(config_path.get());
    auto qwen2vl_tokenizer = mllm::models::qwen2vl::Qwen2VLTokenizer(tokenizer_path.get());
    auto qwen2vl = mllm::models::qwen2vl::Qwen2VLForCausalLM(qwen2vl_cfg);

    auto param = mllm::load(model_dir_path.get());
    qwen2vl.llm.load(param);
    qwen2vl.visual.load(param);

    mllm::print(qwen2vl.llm);
    mllm::print(qwen2vl.visual);
  }

  mllm::memoryReport();
  mllm::shutdownContext();
}
