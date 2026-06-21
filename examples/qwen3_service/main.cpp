#include <string>
#include <vector>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include <mllm/mllm.hpp>
#include <mllm/engine/service/Service.hpp>
#include <mllm/models/qwen3/modeling_qwen3_service.hpp>

MLLM_MAIN({
  mllm::setLogLevel(mllm::LogLevel::kError);
  auto& model_path = mllm::Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  mllm::Argparse::parse(argc, argv);

  auto qwen3_session = std::make_shared<mllm::models::qwen3::Qwen3Session>();
  qwen3_session->fromPreTrain(model_path.get());
  mllm::service::insertSession("mllmTeam/Qwen3-0.6B-w4a32kai", qwen3_session);
  mllm::service::startService();

  std::vector<nlohmann::json> history;
  const std::string model_name = "mllmTeam/Qwen3-0.6B-w4a32kai";

  std::cout << "Enter /exit or /quit to exit this program. /clear for clear context.\n";

  while (true) {
    std::cout << "\nUser: ";
    std::string user_input;
    std::getline(std::cin, user_input);
    if (user_input == "/exit" || user_input == "/quit") break;
    if (user_input == "/clear") {
      history.clear();
      continue;
    }

    nlohmann::json user_msg;
    user_msg["role"] = "user";
    user_msg["content"] = user_input;
    history.push_back(user_msg);

    nlohmann::json req;
    req["model"] = model_name;
    req["messages"] = history;
    req["id"] = "chat-multi";
    req["enable_thinking"] = true;
    mllm::service::sendRequest(req.dump());
    std::string assistant_content;

    bool thinking_states = false;

    while (true) {
      std::string resp = mllm::service::getResponse("chat-multi");
      auto j = nlohmann::json::parse(resp);

      if (j.contains("choices") && j["choices"].size() > 0 && j["choices"][0].contains("delta")
          && j["choices"][0]["delta"].contains("content")) {
        std::string delta = j["choices"][0]["delta"]["content"].get<std::string>();

        if (delta == "<think>") {
          thinking_states = true;
          fmt::print(fmt::fg(fmt::color::gray) | fmt::emphasis::bold | fmt::emphasis::underline, "Thinking...:");
          continue;
        }
        if (delta == "</think>") {
          thinking_states = false;
          fmt::print("\n");
          continue;
        }

        if (thinking_states) {
          fmt::print(fmt::fg(fmt::color::gray), "{}", delta);
          std::fflush(stdout);
        } else {
          fmt::print("{}", delta);
          std::fflush(stdout);
        }

        assistant_content += delta;
      }

      if (j.contains("choices") && j["choices"].size() > 0 && j["choices"][0].contains("finish_reason")
          && j["choices"][0]["finish_reason"].is_string() && j["choices"][0]["finish_reason"].get<std::string>() == "stop") {
        break;
      }
    }

    nlohmann::json assistant_msg;
    assistant_msg["role"] = "assistant";
    assistant_msg["content"] = assistant_content;
    history.push_back(assistant_msg);
  }

  mllm::service::stopService();
  return 0;
})
