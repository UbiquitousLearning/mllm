#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/utils/AnyValue.hpp>
#include <mllm/models/smollm3_3B/tokenization_smollm3.hpp>

using mllm::Argparse;

MLLM_MAIN({
  auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory");
  Argparse::parse(argc, argv);

  {
    auto tokenizer = mllm::models::smollm3::SmolLM3Tokenizer(tokenizer_path.get());
    auto ids = tokenizer.encode(tokenizer.applyChatTemplate("Bonjour 😈", false));
    mllm::print(ids);
    mllm::print(tokenizer.decode(ids));
  }
})
