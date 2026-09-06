#include "mllm/preprocessor/chat_template/ChatTemplate.hpp"

#include <fstream>
#include <map>
#include <sstream>
#include <utility>

#ifndef MLLM_ENABLE_JINJA_CHAT_TEMPLATE
#define MLLM_ENABLE_JINJA_CHAT_TEMPLATE 0
#endif

#if MLLM_ENABLE_JINJA_CHAT_TEMPLATE
#include <jinja.hpp>
#endif

namespace mllm::preprocessor {
namespace {

std::string readFile(const std::filesystem::path& path, std::size_t max_bytes) {
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  if (error) { throw ChatTemplateError("cannot inspect chat template '" + path.string() + "': " + error.message()); }
  if (size > max_bytes) { throw ChatTemplateError("chat template '" + path.string() + "' exceeds the configured size limit"); }

  std::ifstream stream(path, std::ios::binary);
  if (!stream) { throw ChatTemplateError("cannot open chat template '" + path.string() + "'"); }
  std::ostringstream content;
  content << stream.rdbuf();
  if (!stream.good() && !stream.eof()) { throw ChatTemplateError("cannot read chat template '" + path.string() + "'"); }
  return content.str();
}

ChatTemplateSource sourceFromFile(const std::filesystem::path& path, ChatTemplateSourceKind kind, std::size_t max_bytes) {
  return {.kind = kind, .location = path.string(), .content = readFile(path, max_bytes)};
}

bool isRegularFileIfPresent(const std::filesystem::path& path, const char* description) {
  std::error_code error;
  const auto status = std::filesystem::status(path, error);
  if (error == std::make_error_code(std::errc::no_such_file_or_directory)) { return false; }
  if (error) {
    throw ChatTemplateError(std::string("cannot inspect ") + description + " '" + path.string() + "': " + error.message());
  }
  if (!std::filesystem::exists(status)) { return false; }
  if (!std::filesystem::is_regular_file(status)) {
    throw ChatTemplateError(std::string(description) + " '" + path.string() + "' is not a regular file");
  }
  return true;
}

void validateRequest(const ChatTemplateRequest& request) {
  if (!request.messages.is_array()) { throw ChatTemplateError("chat template messages must be a JSON array"); }
  if (request.tools.has_value() && !request.tools->is_array()) {
    throw ChatTemplateError("chat template tools must be a JSON array when provided");
  }
  if (!request.extra_context.is_object()) { throw ChatTemplateError("chat template extra_context must be a JSON object"); }
  if (request.max_rendered_bytes == 0) { throw ChatTemplateError("chat template output limit must be positive"); }
  for (const char* reserved : {"messages", "tools", "add_generation_prompt"}) {
    if (request.extra_context.contains(reserved)) {
      throw ChatTemplateError(std::string("chat template extra_context cannot override reserved key '") + reserved + "'");
    }
  }
}

// Recursively walks a JSON value looking for any control token inside a string.
// `messages` and `tools` are data supplied by the caller and, transitively, by
// the end user; `extra_context` is not scanned because template variables such
// as bos_token are legitimately control tokens.
void rejectControlTokensIn(const nlohmann::ordered_json& value, const std::vector<std::string>& control_tokens,
                           const std::string& path) {
  if (value.is_string()) {
    const auto text = value.get<std::string>();
    for (const auto& token : control_tokens) {
      if (!token.empty() && text.find(token) != std::string::npos) {
        throw ChatTemplateInjectionError("chat template " + path + " contains the control token '" + token
                                         + "'; message and tool content must not open or close a turn");
      }
    }
    return;
  }
  if (value.is_object()) {
    for (const auto& [key, child] : value.items()) { rejectControlTokensIn(child, control_tokens, path + "." + key); }
    return;
  }
  if (value.is_array()) {
    for (std::size_t index = 0; index < value.size(); ++index) {
      rejectControlTokensIn(value[index], control_tokens, path + "[" + std::to_string(index) + "]");
    }
  }
}

void validateRenderedSize(const std::string& output, std::size_t max_bytes, const std::string& location) {
  if (output.size() > max_bytes) {
    throw ChatTemplateError("chat template '" + location + "' exceeded the configured output limit");
  }
}

using NamedTemplateSources = std::map<std::string, ChatTemplateSource>;

ChatTemplateSource selectNamedTemplate(const NamedTemplateSources& templates, const ChatTemplateLoadOptions& options,
                                       const std::string& location) {
  if (options.template_name.has_value()) {
    const auto selected = templates.find(*options.template_name);
    if (selected == templates.end()) {
      throw ChatTemplateError("chat template '" + *options.template_name + "' was not found in '" + location + "'");
    }
    return selected->second;
  }

  if (options.tools_provided) {
    const auto tool_template = templates.find("tool_use");
    if (tool_template != templates.end()) { return tool_template->second; }
  }

  const auto default_template = templates.find("default");
  if (default_template != templates.end()) { return default_template->second; }

  std::string available;
  for (const auto& [name, source] : templates) {
    (void)source;
    if (!available.empty()) { available += ", "; }
    available += name;
  }
  throw ChatTemplateError("multiple chat templates were found in '" + location
                          + "' without a default; select one explicitly (available: " + available + ")");
}

NamedTemplateSources loadStandaloneTemplates(const ChatTemplateLoadOptions& options) {
  NamedTemplateSources templates;
  const auto default_path = options.model_directory / "chat_template.jinja";
  if (isRegularFileIfPresent(default_path, "chat template")) {
    templates.emplace("default",
                      sourceFromFile(default_path, ChatTemplateSourceKind::ModelTemplateFile, options.max_template_bytes));
  }

  const auto additional_directory = options.model_directory / "additional_chat_templates";
  std::error_code error;
  const auto additional_status = std::filesystem::status(additional_directory, error);
  if (error == std::make_error_code(std::errc::no_such_file_or_directory)) { return templates; }
  if (error) {
    throw ChatTemplateError("cannot inspect additional chat templates '" + additional_directory.string()
                            + "': " + error.message());
  }
  if (!std::filesystem::exists(additional_status)) { return templates; }
  if (!std::filesystem::is_directory(additional_status)) {
    throw ChatTemplateError("additional chat templates '" + additional_directory.string() + "' is not a directory");
  }

  std::filesystem::directory_iterator iterator(additional_directory, error);
  if (error) {
    throw ChatTemplateError("cannot list additional chat templates '" + additional_directory.string()
                            + "': " + error.message());
  }
  for (const auto end = std::filesystem::directory_iterator(); iterator != end; iterator.increment(error)) {
    if (error) {
      throw ChatTemplateError("cannot list additional chat templates '" + additional_directory.string()
                              + "': " + error.message());
    }
    const auto& entry = *iterator;
    if (entry.path().extension() != ".jinja") { continue; }
    if (!entry.is_regular_file(error)) {
      if (error) {
        throw ChatTemplateError("cannot inspect additional chat template '" + entry.path().string() + "': " + error.message());
      }
      throw ChatTemplateError("additional chat template '" + entry.path().string() + "' is not a regular file");
    }
    templates[entry.path().stem().string()] =
        sourceFromFile(entry.path(), ChatTemplateSourceKind::AdditionalTemplateFile, options.max_template_bytes);
  }
  if (error) {
    throw ChatTemplateError("cannot list additional chat templates '" + additional_directory.string()
                            + "': " + error.message());
  }
  return templates;
}

nlohmann::ordered_json parseTokenizerConfig(const std::filesystem::path& tokenizer_config_path, std::size_t max_bytes) {
  const auto config_text = readFile(tokenizer_config_path, max_bytes);
  try {
    auto config = nlohmann::ordered_json::parse(config_text);
    if (!config.is_object()) { throw ChatTemplateError("tokenizer config '" + tokenizer_config_path.string() + "' is not a JSON object"); }
    return config;
  } catch (const ChatTemplateError&) { throw; } catch (const std::exception& parse_error) {
    throw ChatTemplateError("cannot parse tokenizer config '" + tokenizer_config_path.string() + "': " + parse_error.what());
  }
}

NamedTemplateSources parseNamedConfigTemplates(const nlohmann::ordered_json& configured_template,
                                               const std::filesystem::path& tokenizer_config_path) {
  NamedTemplateSources templates;
  const auto add_template = [&](const std::string& name, const std::string& content) {
    templates[name] = ChatTemplateSource{.kind = ChatTemplateSourceKind::TokenizerConfig,
                                         .location = tokenizer_config_path.string() + "#chat_template." + name,
                                         .content = content};
  };

  if (configured_template.is_object()) {
    for (const auto& [name, content] : configured_template.items()) {
      if (!content.is_string()) {
        throw ChatTemplateError("unsupported named chat_template value in '" + tokenizer_config_path.string()
                                + "'; every template must be a string");
      }
      add_template(name, content.get<std::string>());
    }
    return templates;
  }

  if (configured_template.is_array()) {
    for (const auto& entry : configured_template) {
      if (!entry.is_object() || !entry.contains("name") || !entry.at("name").is_string() || !entry.contains("template")
          || !entry.at("template").is_string()) {
        throw ChatTemplateError("unsupported named chat_template entry in '" + tokenizer_config_path.string()
                                + "'; expected objects containing string name and template fields");
      }
      add_template(entry.at("name").get<std::string>(), entry.at("template").get<std::string>());
    }
    return templates;
  }

  return templates;
}

}  // namespace

class JinjaChatTemplate::Impl {
 public:
  Impl(ChatTemplateSource source, nlohmann::ordered_json default_context)
      : source_(std::move(source)), default_context_(std::move(default_context)) {
    if (!default_context_.is_object()) { throw ChatTemplateError("chat template default context must be a JSON object"); }
#if MLLM_ENABLE_JINJA_CHAT_TEMPLATE
    template_ = std::make_unique<jinja::Template>(source_.content);
    template_->add_function("raise_exception", [location = source_.location](const std::vector<jinja::Argument>& args) {
      std::string message = "template requested an exception";
      if (!args.empty() && args.front().second.is_string()) { message = args.front().second.get<std::string>(); }
      throw ChatTemplateError("chat template '" + location + "': " + message);
      return jinja::json();
    });
#else
    throw ChatTemplateError("chat template '" + source_.location
                            + "' requires Jinja support, but this binary was built with "
                              "MLLM_ENABLE_JINJA_CHAT_TEMPLATE=OFF");
#endif
  }

  std::string render(const ChatTemplateRequest& request) const {
#if MLLM_ENABLE_JINJA_CHAT_TEMPLATE
    nlohmann::ordered_json context = default_context_;
    for (const auto& [key, value] : request.extra_context.items()) { context[key] = value; }
    context["messages"] = request.messages;
    context["add_generation_prompt"] = request.add_generation_prompt;
    if (request.tools.has_value()) { context["tools"] = *request.tools; }
    return template_->render(jinja::json(std::move(context)));
#else
    (void)request;
    throw ChatTemplateError("Jinja chat-template rendering is unavailable in this binary");
#endif
  }

  ChatTemplateSource source_;
  nlohmann::ordered_json default_context_;
#if MLLM_ENABLE_JINJA_CHAT_TEMPLATE
  std::unique_ptr<jinja::Template> template_;
#endif
};

JinjaChatTemplate::JinjaChatTemplate(ChatTemplateSource source, nlohmann::ordered_json default_context) {
  try {
    impl_ = std::make_unique<Impl>(std::move(source), std::move(default_context));
  } catch (const ChatTemplateError&) { throw; } catch (const std::exception& error) {
    throw ChatTemplateError(std::string("failed to compile chat template: ") + error.what());
  }
}

JinjaChatTemplate::~JinjaChatTemplate() = default;
JinjaChatTemplate::JinjaChatTemplate(JinjaChatTemplate&&) noexcept = default;
JinjaChatTemplate& JinjaChatTemplate::operator=(JinjaChatTemplate&&) noexcept = default;

std::string JinjaChatTemplate::render(const ChatTemplateRequest& request) const {
  validateRequest(request);
  try {
    auto output = impl_->render(request);
    validateRenderedSize(output, request.max_rendered_bytes, impl_->source_.location);
    return output;
  } catch (const ChatTemplateError&) { throw; } catch (const std::exception& error) {
    throw ChatTemplateError("failed to render chat template '" + impl_->source_.location + "': " + error.what());
  }
}

const ChatTemplateSource& JinjaChatTemplate::source() const noexcept { return impl_->source_; }

class ChatPreprocessor::Impl {
 public:
  Impl(ChatPreprocessorConfig config, LegacyChatTemplateRenderer legacy_renderer)
      : backend_(config.backend), legacy_renderer_(std::move(legacy_renderer)), control_tokens_(config.control_tokens) {
    switch (backend_) {
      case ChatTemplateBackend::Legacy:
        if (!legacy_renderer_) { throw ChatTemplateError("legacy chat-template backend requires an explicit renderer"); }
        return;
      case ChatTemplateBackend::JinjaRequired: {
        if (!jinjaChatTemplatesAvailable()) {
          throw ChatTemplateError("JinjaRequired chat-template backend was selected, but this binary was built with "
                                  "MLLM_ENABLE_JINJA_CHAT_TEMPLATE=OFF");
        }
        auto source = ChatTemplateLoader::find(config.template_options);
        if (!source.has_value()) {
          const auto location = config.template_options.model_directory.empty()
                                    ? std::string("the configured template source")
                                    : "model directory '" + config.template_options.model_directory.string() + "'";
          throw ChatTemplateError("JinjaRequired chat-template backend found no template in " + location);
        }
        const auto special_tokens = ChatTemplateLoader::specialTokenContext(config.template_options);
        jinja_template_ = std::make_unique<JinjaChatTemplate>(std::move(*source), special_tokens);
        if (!config.template_options.tools_provided && !config.template_options.template_name.has_value()
            && !config.template_options.explicit_template_path.has_value()) {
          auto tool_options = config.template_options;
          tool_options.tools_provided = true;
          auto tool_source = ChatTemplateLoader::find(tool_options);
          if (tool_source.has_value()
              && (tool_source->kind != jinja_template_->source().kind
                  || tool_source->location != jinja_template_->source().location
                  || tool_source->content != jinja_template_->source().content)) {
            tool_jinja_template_ = std::make_unique<JinjaChatTemplate>(std::move(*tool_source), special_tokens);
          }
        }
        return;
      }
    }
    throw ChatTemplateError("unknown chat-template backend");
  }

  std::string render(const ChatTemplateRequest& request) const {
    validateRequest(request);
    // Enforced for both backends: a legacy renderer copies content through just
    // as literally as an official template does.
    rejectControlTokensInContent(request, control_tokens_);
    if (backend_ == ChatTemplateBackend::JinjaRequired) {
      if (request.tools.has_value() && tool_jinja_template_) { return tool_jinja_template_->render(request); }
      return jinja_template_->render(request);
    }

    try {
      auto output = legacy_renderer_(request);
      validateRenderedSize(output, request.max_rendered_bytes, "legacy renderer");
      return output;
    } catch (const ChatTemplateError&) { throw; } catch (const std::exception& error) {
      throw ChatTemplateError(std::string("legacy chat-template renderer failed: ") + error.what());
    }
  }

  ChatTemplateBackend backend_;
  LegacyChatTemplateRenderer legacy_renderer_;
  std::vector<std::string> control_tokens_;
  std::unique_ptr<JinjaChatTemplate> jinja_template_;
  std::unique_ptr<JinjaChatTemplate> tool_jinja_template_;
};

ChatPreprocessor::ChatPreprocessor(ChatPreprocessorConfig config, LegacyChatTemplateRenderer legacy_renderer)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(legacy_renderer))) {}

ChatPreprocessor::~ChatPreprocessor() = default;
ChatPreprocessor::ChatPreprocessor(ChatPreprocessor&&) noexcept = default;
ChatPreprocessor& ChatPreprocessor::operator=(ChatPreprocessor&&) noexcept = default;

std::string ChatPreprocessor::render(const ChatTemplateRequest& request) const { return impl_->render(request); }

ChatTemplateBackend ChatPreprocessor::backend() const noexcept { return impl_->backend_; }

void ChatPreprocessor::setControlTokens(std::vector<std::string> control_tokens) {
  impl_->control_tokens_ = std::move(control_tokens);
}

const ChatTemplateSource* ChatPreprocessor::source() const noexcept {
  return impl_->jinja_template_ ? &impl_->jinja_template_->source() : nullptr;
}

nlohmann::ordered_json ChatTemplateLoader::specialTokenContext(const ChatTemplateLoadOptions& options) {
  if (options.special_tokens.has_value()) {
    if (!options.special_tokens->is_object()) { throw ChatTemplateError("chat template special_tokens must be a JSON object"); }
    return *options.special_tokens;
  }
  nlohmann::ordered_json context = nlohmann::ordered_json::object();
  if (options.model_directory.empty()) { return context; }
  const auto tokenizer_config_path = options.model_directory / "tokenizer_config.json";
  if (!isRegularFileIfPresent(tokenizer_config_path, "tokenizer config")) { return context; }
  const auto config = parseTokenizerConfig(tokenizer_config_path, options.max_template_bytes);
  // Mirrors the special tokens Transformers exposes as template variables.
  for (const char* name : {"bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"}) {
    if (!config.contains(name)) { continue; }
    const auto& token = config.at(name);
    if (token.is_string()) {
      context[name] = token.get<std::string>();
    } else if (token.is_object() && token.contains("content") && token.at("content").is_string()) {
      context[name] = token.at("content").get<std::string>();
    } else if (token.is_null()) {
      context[name] = nullptr;
    } else {
      throw ChatTemplateError(std::string("unsupported ") + name + " value in '" + tokenizer_config_path.string()
                              + "'; expected a string or an object with string content");
    }
  }
  return context;
}

std::optional<ChatTemplateSource> ChatTemplateLoader::find(const ChatTemplateLoadOptions& options) {
  if (options.explicit_template_path.has_value()) {
    return sourceFromFile(*options.explicit_template_path, ChatTemplateSourceKind::ExplicitFile, options.max_template_bytes);
  }

  if (options.model_directory.empty()) { return std::nullopt; }

  auto standalone_templates = loadStandaloneTemplates(options);
  if (!standalone_templates.empty()) {
    return selectNamedTemplate(standalone_templates, options, options.model_directory.string());
  }

  const auto tokenizer_config_path = options.model_directory / "tokenizer_config.json";
  if (!isRegularFileIfPresent(tokenizer_config_path, "tokenizer config")) { return std::nullopt; }

  const auto config = parseTokenizerConfig(tokenizer_config_path, options.max_template_bytes);
  if (!config.contains("chat_template")) { return std::nullopt; }
  const auto& configured_template = config.at("chat_template");
  if (configured_template.is_string()) {
    if (options.template_name.has_value()) {
      throw ChatTemplateError("cannot select named chat template '" + *options.template_name + "' from the single template in '"
                              + tokenizer_config_path.string() + "'");
    }
    return ChatTemplateSource{.kind = ChatTemplateSourceKind::TokenizerConfig,
                              .location = tokenizer_config_path.string() + "#chat_template",
                              .content = configured_template.get<std::string>()};
  }

  auto named_templates = parseNamedConfigTemplates(configured_template, tokenizer_config_path);
  if (!named_templates.empty()) {
    return selectNamedTemplate(named_templates, options, tokenizer_config_path.string() + "#chat_template");
  }
  throw ChatTemplateError("unsupported chat_template value in '" + tokenizer_config_path.string()
                          + "'; expected a string or a collection of named templates");
}

void rejectControlTokensInContent(const ChatTemplateRequest& request, const std::vector<std::string>& control_tokens) {
  if (control_tokens.empty()) { return; }
  rejectControlTokensIn(request.messages, control_tokens, "messages");
  if (request.tools.has_value()) { rejectControlTokensIn(*request.tools, control_tokens, "tools"); }
}

bool jinjaChatTemplatesAvailable() noexcept {
#if MLLM_ENABLE_JINJA_CHAT_TEMPLATE
  return true;
#else
  return false;
#endif
}

ChatTemplateBackend parseChatTemplateBackend(std::string_view name) {
  if (name == "legacy") { return ChatTemplateBackend::Legacy; }
  if (name == "jinja_required") { return ChatTemplateBackend::JinjaRequired; }
  throw ChatTemplateError("unsupported chat-template backend '" + std::string(name)
                          + "'; expected 'legacy' or 'jinja_required'");
}

const char* chatTemplateBackendName(ChatTemplateBackend backend) noexcept {
  switch (backend) {
    case ChatTemplateBackend::Legacy: return "legacy";
    case ChatTemplateBackend::JinjaRequired: return "jinja_required";
  }
  return "unknown";
}

}  // namespace mllm::preprocessor
