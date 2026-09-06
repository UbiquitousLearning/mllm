#include <fstream>
#include <iostream>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "mllm/preprocessor/chat_template/ChatTemplate.hpp"

namespace {

nlohmann::ordered_json loadRequest(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) { throw std::runtime_error("cannot open render request '" + path + "'"); }
  return nlohmann::ordered_json::parse(stream);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: Mllm-ChatTemplate-Render <chat_template.jinja | model directory> <request.json>\n";
    return 2;
  }

  try {
    const auto request = loadRequest(argv[2]);
    if (!request.is_object() || !request.contains("messages")) {
      throw std::invalid_argument("render request must be an object containing messages");
    }

    // The template's directory doubles as the model directory so special-token
    // variables (bos_token, ...) come from its tokenizer_config.json, exactly as
    // the product path sees them.
    // A directory uses the product discovery order (chat_template.jinja,
    // additional_chat_templates/, tokenizer_config.json); a file is explicit.
    const std::filesystem::path template_path = std::filesystem::absolute(argv[1]);
    mllm::preprocessor::ChatTemplateLoadOptions load_options;
    if (std::filesystem::is_directory(template_path)) {
      load_options.model_directory = template_path;
    } else {
      load_options.model_directory = template_path.parent_path();
      load_options.explicit_template_path = template_path;
    }
    mllm::preprocessor::ChatPreprocessor chat_preprocessor(
        {.backend = mllm::preprocessor::ChatTemplateBackend::JinjaRequired, .template_options = load_options});

    mllm::preprocessor::ChatTemplateRequest render_request;
    render_request.messages = request.at("messages");
    render_request.add_generation_prompt = request.value("add_generation_prompt", true);
    if (request.contains("tools")) { render_request.tools = request.at("tools"); }
    if (request.contains("extra_context")) { render_request.extra_context = request.at("extra_context"); }

    const auto output = chat_preprocessor.render(render_request);
    std::cout.write(output.data(), static_cast<std::streamsize>(output.size()));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
