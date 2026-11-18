// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Chat-template probe for parity gates. Two modes:
//   --tokenize            tokenize stdin with the model tokenizer and print ids
// Models: qwen3_5, minicpm5, qwen3, qwen3_moe, qwen_ascend, minicpm4, qwen_npu (needs --merges)
//   (default)             run the product path (tokenizer.convertMessage) with an
//                         explicit chat-template backend and print the rendered
//                         prompt plus token ids
// Output is one JSON object on stdout.
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "mllm/mllm.hpp"
#include "mllm/models/minicpm4/configuration_minicpm4.hpp"
#include "mllm/models/minicpm4/tokenization_minicpm4.hpp"
#include "mllm/models/minicpm5/configuration_minicpm5.hpp"
#include "mllm/models/minicpm5/tokenization_minicpm5.hpp"
#include "mllm/models/qwen3/configuration_qwen3.hpp"
#include "mllm/models/qwen3/tokenization_qwen3.hpp"
#include "mllm/models/qwen3_5/configuration_qwen3_5.hpp"
#include "mllm/models/qwen3_5/tokenization_qwen3_5.hpp"
#include "mllm/models/qwen3_moe/configuration_qwen3_moe.hpp"
#include "mllm/models/qwen3_moe/tokenization_qwen3_moe.hpp"
#include "mllm/models/qwen_ascend/configuration_qwen_ascend.hpp"
#include "mllm/models/qwen_ascend/tokenization_qwen_ascend.hpp"
#include "mllm/models/qwen_npu/configuration_qwen_npu.hpp"
#include "mllm/models/qwen_npu/tokenization_qwen.hpp"
#include "mllm/preprocessor/chat_template/ChatTemplate.hpp"

namespace {

struct Options {
  std::string model;
  std::string tokenizer;
  std::string merges;
  std::string config;
  std::string model_dir;
  std::optional<std::string> backend;
  std::string prompt;
  std::string system;
  bool enable_thinking = false;
  bool tokenize = false;
};

[[noreturn]] void usage(const std::string& error = "") {
  if (!error.empty()) { std::cerr << error << "\n"; }
  std::cerr << "usage: Mllm-ChatTemplate-Probe --model MODEL --tokenizer tokenizer.json [--merges merges.txt] [--config config.json]\n"
               "       [--model_dir DIR] [--backend legacy|jinja_required] [--prompt TEXT] [--system TEXT]\n"
               "       [--enable_thinking] [--tokenize]\n";
  std::exit(2);
}

Options parse(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) { usage("missing value for " + arg); }
      return argv[++i];
    };
    if (arg == "--model") {
      options.model = value();
    } else if (arg == "--tokenizer") {
      options.tokenizer = value();
    } else if (arg == "--merges") {
      options.merges = value();
    } else if (arg == "--config") {
      options.config = value();
    } else if (arg == "--model_dir") {
      options.model_dir = value();
    } else if (arg == "--backend") {
      options.backend = value();
    } else if (arg == "--prompt") {
      options.prompt = value();
    } else if (arg == "--system") {
      options.system = value();
    } else if (arg == "--enable_thinking") {
      options.enable_thinking = true;
    } else if (arg == "--tokenize") {
      options.tokenize = true;
    } else {
      usage("unknown argument " + arg);
    }
  }
  static const char* kModels[] = {"qwen3_5", "minicpm5", "qwen3", "qwen3_moe", "qwen_ascend", "minicpm4", "qwen_npu"};
  bool known = false;
  for (const char* model : kModels) { known = known || options.model == model; }
  if (!known) { usage("unknown --model " + options.model); }
  if (options.model == "qwen_npu" && options.merges.empty()) { usage("--merges is required for qwen_npu"); }
  if (options.tokenizer.empty()) { usage("--tokenizer is required"); }
  if (!options.tokenize && options.config.empty()) { usage("--config is required for the product path"); }
  if (!options.tokenize && options.prompt.empty()) { usage("--prompt is required for the product path"); }
  return options;
}

std::vector<int64_t> toVector(const mllm::Tensor& tensor) {
  return {tensor.ptr<int64_t>(), tensor.ptr<int64_t>() + tensor.numel()};
}

template<typename Tokenizer>
std::vector<int64_t> tokenizeStdin(Tokenizer& tokenizer) {
  const std::string text((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
  if constexpr (requires { tokenizer.encode(text); }) {
    return tokenizer.encode(text);  // UTF-8 tokenizers (MiniCPM4)
  } else {
    return toVector(tokenizer.convert2Ids(tokenizer.tokenize(text)));
  }
}

// All runners use the same tokenize/product paths; only the message fields and
// tokenizer inputs (an extra merges file for Qwen NPU) vary by model.
template<typename Tokenizer, typename Config, typename Message, typename... Files>
void runModel(const Options& options, nlohmann::json& output, const Message& message, const Files&... extra_files) {
  if (options.tokenize) {
    Tokenizer tokenizer(options.tokenizer, extra_files...);
    output["token_ids"] = tokenizeStdin(tokenizer);
    return;
  }

  Config cfg(options.config);
  if (options.backend) { cfg.chat_template_backend = mllm::preprocessor::parseChatTemplateBackend(*options.backend); }
  Tokenizer tokenizer(options.tokenizer, extra_files..., cfg, options.model_dir);
  output["backend"] = mllm::preprocessor::chatTemplateBackendName(tokenizer.chatTemplateBackend());
  output["prompt_text"] = tokenizer.renderChatTemplate(message);
  output["token_ids"] = toVector(tokenizer.convertMessage(message).at("sequence"));
}

}  // namespace

int main(int argc, char** argv) {
  const auto options = parse(argc, argv);
  try {
    mllm::initializeContext();
    nlohmann::json output;
    output["model"] = options.model;
    output["jinja_available"] = mllm::preprocessor::jinjaChatTemplatesAvailable();

    if (options.model == "qwen3_5") {
      using namespace mllm::models::qwen3_5;
      runModel<Qwen3_5Tokenizer, Qwen3_5Config>(options, output, Qwen3_5Message{.prompt = options.prompt});
    } else if (options.model == "qwen3") {
      using namespace mllm::models::qwen3;
      runModel<Qwen3Tokenizer, Qwen3Config>(options, output, Qwen3Message{.prompt = options.prompt});
    } else if (options.model == "qwen3_moe") {
      using namespace mllm::models::qwen3_moe;
      runModel<Qwen3Tokenizer, Qwen3MoeConfig>(options, output, Qwen3Message{.prompt = options.prompt});
    } else if (options.model == "qwen_ascend") {
      using namespace mllm::models::qwen_ascend;
      runModel<QwenAscendTokenizer, QwenAscendConfig>(options, output, QwenAscendMessage{.prompt = options.prompt});
    } else if (options.model == "minicpm4") {
      using namespace mllm::models::minicpm4;
      runModel<MiniCPM4Tokenizer, MiniCPM4Config>(options, output, MiniCPM4Message{.prompt = options.prompt});
    } else if (options.model == "qwen_npu") {
      using namespace mllm::models::qwen_npu;
      runModel<QwenTokenizer, QwenNPUConfig>(options, output, QwenMessage{.prompt = options.prompt}, options.merges);
    } else {
      using namespace mllm::models::minicpm5;
      runModel<MiniCPM5Tokenizer, MiniCPM5Config>(
          options, output, MiniCPM5Message{.prompt = options.prompt, .system = options.system, .enable_thinking = options.enable_thinking});
    }
    std::cout << output.dump() << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
