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
#include <memory>
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

// Text-only runners: tokenize mode or the product path with an explicit backend.
template<typename Tokenizer, typename Config, typename Message, typename MakeTokenizer>
void runTextModel(const Options& options, nlohmann::json& output, MakeTokenizer make_tokenizer) {
  if (options.tokenize) {
    auto tokenizer = make_tokenizer(std::optional<Config>{});
    output["token_ids"] = tokenizeStdin(*tokenizer);
    return;
  }
  Config cfg(options.config);
  if (options.backend) { cfg.chat_template_backend = mllm::preprocessor::parseChatTemplateBackend(*options.backend); }
  auto tokenizer = make_tokenizer(std::optional<Config>{cfg});
  Message message{.prompt = options.prompt};
  output["backend"] = mllm::preprocessor::chatTemplateBackendName(tokenizer->chatTemplateBackend());
  output["prompt_text"] = tokenizer->renderChatTemplate(message);
  output["token_ids"] = toVector(tokenizer->convertMessage(message).at("sequence"));
}

}  // namespace

int main(int argc, char** argv) {
  const auto options = parse(argc, argv);
  try {
    mllm::initializeContext();
    nlohmann::json output;
    output["model"] = options.model;
    output["jinja_available"] = mllm::preprocessor::jinjaChatTemplatesAvailable();
    const std::filesystem::path model_dir = options.model_dir;

    if (options.model == "qwen3_5") {
      if (options.tokenize) {
        mllm::models::qwen3_5::Qwen3_5Tokenizer tokenizer(options.tokenizer);
        output["token_ids"] = tokenizeStdin(tokenizer);
      } else {
        auto cfg = mllm::models::qwen3_5::Qwen3_5Config(options.config);
        if (options.backend) { cfg.chat_template_backend = mllm::preprocessor::parseChatTemplateBackend(*options.backend); }
        mllm::models::qwen3_5::Qwen3_5Tokenizer tokenizer(options.tokenizer, cfg, model_dir);
        mllm::models::qwen3_5::Qwen3_5Message message{.prompt = options.prompt};
        output["backend"] = mllm::preprocessor::chatTemplateBackendName(tokenizer.chatTemplateBackend());
        output["prompt_text"] = tokenizer.renderChatTemplate(message);
        output["token_ids"] = toVector(tokenizer.convertMessage(message).at("sequence"));
      }
    } else if (options.model == "qwen3") {
      using namespace mllm::models::qwen3;
      runTextModel<Qwen3Tokenizer, Qwen3Config, Qwen3Message>(options, output, [&](const std::optional<Qwen3Config>& cfg) {
        return cfg ? std::make_unique<Qwen3Tokenizer>(options.tokenizer, *cfg, model_dir) : std::make_unique<Qwen3Tokenizer>(options.tokenizer);
      });
    } else if (options.model == "qwen3_moe") {
      using namespace mllm::models::qwen3_moe;
      runTextModel<Qwen3Tokenizer, Qwen3MoeConfig, Qwen3Message>(options, output, [&](const std::optional<Qwen3MoeConfig>& cfg) {
        return cfg ? std::make_unique<Qwen3Tokenizer>(options.tokenizer, *cfg, model_dir) : std::make_unique<Qwen3Tokenizer>(options.tokenizer);
      });
    } else if (options.model == "qwen_ascend") {
      using namespace mllm::models::qwen_ascend;
      runTextModel<QwenAscendTokenizer, QwenAscendConfig, QwenAscendMessage>(
          options, output, [&](const std::optional<QwenAscendConfig>& cfg) {
            return cfg ? std::make_unique<QwenAscendTokenizer>(options.tokenizer, *cfg, model_dir)
                       : std::make_unique<QwenAscendTokenizer>(options.tokenizer);
          });
    } else if (options.model == "minicpm4") {
      using namespace mllm::models::minicpm4;
      runTextModel<MiniCPM4Tokenizer, MiniCPM4Config, MiniCPM4Message>(options, output, [&](const std::optional<MiniCPM4Config>& cfg) {
        return cfg ? std::make_unique<MiniCPM4Tokenizer>(options.tokenizer, *cfg, model_dir)
                   : std::make_unique<MiniCPM4Tokenizer>(options.tokenizer);
      });
    } else if (options.model == "qwen_npu") {
      using namespace mllm::models::qwen_npu;
      runTextModel<QwenTokenizer, QwenNPUConfig, QwenMessage>(options, output, [&](const std::optional<QwenNPUConfig>& cfg) {
        return cfg ? std::make_unique<QwenTokenizer>(options.tokenizer, options.merges, *cfg, model_dir)
                   : std::make_unique<QwenTokenizer>(options.tokenizer, options.merges);
      });
    } else {
      if (options.tokenize) {
        mllm::models::minicpm5::MiniCPM5Tokenizer tokenizer(options.tokenizer);
        output["token_ids"] = tokenizeStdin(tokenizer);
      } else {
        auto cfg = mllm::models::minicpm5::MiniCPM5Config(options.config);
        if (options.backend) { cfg.chat_template_backend = mllm::preprocessor::parseChatTemplateBackend(*options.backend); }
        mllm::models::minicpm5::MiniCPM5Tokenizer tokenizer(options.tokenizer, cfg, model_dir);
        mllm::models::minicpm5::MiniCPM5Message message{
            .prompt = options.prompt, .system = options.system, .enable_thinking = options.enable_thinking};
        output["backend"] = mllm::preprocessor::chatTemplateBackendName(tokenizer.chatTemplateBackend());
        output["prompt_text"] = tokenizer.renderChatTemplate(message);
        output["token_ids"] = toVector(tokenizer.convertMessage(message).at("sequence"));
      }
    }
    std::cout << output.dump() << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
