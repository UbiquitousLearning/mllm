#ifndef MLLM_PREPROCESSOR_CHAT_TEMPLATE_CHAT_TEMPLATE_HPP
#define MLLM_PREPROCESSOR_CHAT_TEMPLATE_CHAT_TEMPLATE_HPP

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace mllm::preprocessor {

enum class ChatTemplateBackend {
  Legacy,
  JinjaRequired,
};

enum class ChatTemplateSourceKind {
  ExplicitFile,
  ModelTemplateFile,
  AdditionalTemplateFile,
  TokenizerConfig,
  InMemory,
};

struct ChatTemplateSource {
  ChatTemplateSourceKind kind = ChatTemplateSourceKind::InMemory;
  std::string location;
  std::string content;
};

struct ChatTemplateRequest {
  nlohmann::ordered_json messages = nlohmann::ordered_json::array();
  bool add_generation_prompt = true;
  std::optional<nlohmann::ordered_json> tools;
  nlohmann::ordered_json extra_context = nlohmann::ordered_json::object();
  std::size_t max_rendered_bytes = 16 * 1024 * 1024;
};

struct ChatTemplateLoadOptions {
  std::filesystem::path model_directory;
  std::optional<std::filesystem::path> explicit_template_path;
  std::optional<std::string> template_name;
  bool tools_provided = false;
  std::size_t max_template_bytes = 1024 * 1024;
  // Template variables that Transformers supplies from the tokenizer, such as
  // bos_token and eos_token. When absent they are read from
  // <model directory>/tokenizer_config.json; request extra_context overrides.
  std::optional<nlohmann::ordered_json> special_tokens;
};

struct ChatPreprocessorConfig {
  ChatTemplateBackend backend = ChatTemplateBackend::Legacy;
  ChatTemplateLoadOptions template_options;
  // The checkpoint's control tokens, normally `BPE::controlTokens()`. Message
  // and tool content carrying one of these is rejected before rendering, so a
  // prompt cannot forge a turn boundary. Leaving this empty disables the check
  // and is only appropriate for a caller that has already sanitized its input.
  std::vector<std::string> control_tokens;
};

class ChatTemplateError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Raised when message or tool content contains one of the checkpoint's control
// tokens. Rendering such a request would let the content open or close a turn,
// so it is refused rather than escaped.
class ChatTemplateInjectionError : public ChatTemplateError {
 public:
  using ChatTemplateError::ChatTemplateError;
};

using LegacyChatTemplateRenderer = std::function<std::string(const ChatTemplateRequest&)>;

// Compiles and renders one Jinja template. Construction fails when Jinja
// support is not present in the current binary.
class JinjaChatTemplate final {
 public:
  explicit JinjaChatTemplate(ChatTemplateSource source,
                             nlohmann::ordered_json default_context = nlohmann::ordered_json::object());
  ~JinjaChatTemplate();

  JinjaChatTemplate(JinjaChatTemplate&&) noexcept;
  JinjaChatTemplate& operator=(JinjaChatTemplate&&) noexcept;
  JinjaChatTemplate(const JinjaChatTemplate&) = delete;
  JinjaChatTemplate& operator=(const JinjaChatTemplate&) = delete;

  std::string render(const ChatTemplateRequest& request) const;
  const ChatTemplateSource& source() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Model-facing entry point. The model selects the backend explicitly; the
// build option only controls whether JinjaRequired can be constructed.
class ChatPreprocessor final {
 public:
  explicit ChatPreprocessor(ChatPreprocessorConfig config, LegacyChatTemplateRenderer legacy_renderer = {});
  ~ChatPreprocessor();

  ChatPreprocessor(ChatPreprocessor&&) noexcept;
  ChatPreprocessor& operator=(ChatPreprocessor&&) noexcept;
  ChatPreprocessor(const ChatPreprocessor&) = delete;
  ChatPreprocessor& operator=(const ChatPreprocessor&) = delete;

  std::string render(const ChatTemplateRequest& request) const;
  ChatTemplateBackend backend() const noexcept;
  const ChatTemplateSource* source() const noexcept;

  // Supplies the checkpoint's control tokens after construction. A tokenizer
  // loads its vocabulary in its constructor body, so it cannot pass them
  // through ChatPreprocessorConfig. Until this is called the injection check is
  // inactive, so every model that owns a vocabulary must call it.
  void setControlTokens(std::vector<std::string> control_tokens);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class ChatTemplateLoader final {
 public:
  static std::optional<ChatTemplateSource> find(const ChatTemplateLoadOptions& options);
  // Reads the special-token template variables (bos_token, eos_token, ...)
  // from <model directory>/tokenizer_config.json. Returns an empty object when
  // the file or the fields are absent.
  static nlohmann::ordered_json specialTokenContext(const ChatTemplateLoadOptions& options);
};

// Throws ChatTemplateInjectionError when any string inside `messages` or
// `tools` contains one of `control_tokens`. Exposed for tests and for models
// that build a prompt outside ChatPreprocessor.
void rejectControlTokensInContent(const ChatTemplateRequest& request, const std::vector<std::string>& control_tokens);

bool jinjaChatTemplatesAvailable() noexcept;
ChatTemplateBackend parseChatTemplateBackend(std::string_view name);
const char* chatTemplateBackendName(ChatTemplateBackend backend) noexcept;

}  // namespace mllm::preprocessor

#endif  // MLLM_PREPROCESSOR_CHAT_TEMPLATE_CHAT_TEMPLATE_HPP
