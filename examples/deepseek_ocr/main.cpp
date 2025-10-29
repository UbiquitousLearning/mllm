#include <mllm/mllm.hpp>
#include <mllm/models/deepseek_ocr/modeling_deepseek_ocr.hpp>
#include <mllm/models/deepseek_ocr/tokenization_deepseek_ocr.hpp>

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory").required(true);
  auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);

  Argparse::parse(argc, argv);
  mllm::setLogLevel(mllm::LogLevel::kError);

  auto config = mllm::models::deepseek_ocr::DpskOcrConfig(config_path.get());
  auto model = mllm::models::deepseek_ocr::DeepseekOCRForCausalLM(config);
  auto tokenizer = mllm::models::deepseek_ocr::DpskOcrTokenizer(tokenizer_path.get());
  model.load(mllm::load(model_path.get(), mllm::ModelFileVersion::kV2));

  model.infer(tokenizer, "<image>\n<|grounding|>Convert the document to markdown. ", "dpsk-ocr-640-640.png", ".", 512);
});
