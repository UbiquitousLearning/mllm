#include <string>
#include <vector>
#include <filesystem>
#include <iostream>
#include <sstream>

#include <fmt/core.h>
#include <fmt/color.h>
#include <nlohmann/json.hpp>

#include <mllm/mllm.hpp>
#include <mllm/engine/service/Service.hpp>
#include "mllm/models/qwen3/modeling_qwen3_probing_service.hpp"

using namespace mllm;
using namespace mllm::models::qwen3_probing;
namespace fs = std::filesystem;

std::vector<int> parseLayers(const std::string& input) {
  std::vector<int> layers;
  std::stringstream ss(input);
  std::string segment;
  int num;
  ss >> segment;
  while (ss >> num) layers.push_back(num);
  return layers;
}

MLLM_MAIN({
  mllm::setLogLevel(mllm::LogLevel::kError);
  auto& model_path = mllm::Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& probe_path = mllm::Argparse::add<std::string>("-p|--probe_path").help("Probes dir").required(true);
  mllm::Argparse::parse(argc, argv);

  auto qwen3_session = std::make_shared<Qwen3ProbingSession>();
  try {
    qwen3_session->fromPreTrain(model_path.get());
  } catch (const std::exception& e) {
    std::cerr << "Load Model Error: " << e.what() << std::endl;
    return 1;
  }

  ProbingArgs p_args;
  p_args.enable_prefill_check = true;
  p_args.enable_decode_check = true;

  p_args.prefill_stop_threshold = 0.7f;
  p_args.decode_stop_threshold = 0.8f;

  p_args.pos_threshold = 0.9f;

  std::cout << ">>> Loading Probes..." << std::endl;
  qwen3_session->setProbingArgs(p_args);
  qwen3_session->loadProbes(probe_path.get(), p_args);

  mllm::service::insertSession("mllmTeam/Qwen3-Probing", qwen3_session);
  mllm::service::startService();

  std::vector<nlohmann::json> history;
  std::vector<int> current_prefill_layers = {27, 30};  // 默认

  std::cout << "\n[System] Ready. Commands:\n";
  std::cout << "  /prefill 15 20   Set prefill layers\n";
  std::cout << "  /clear           Clear history\n";
  std::cout << "  /exit            Exit\n";

  while (true) {
    std::cout << "\nUser: ";
    std::string user_input;
    std::getline(std::cin, user_input);

    if (user_input == "/exit") break;
    if (user_input == "/clear") {
      history.clear();
      continue;
    }
    if (user_input.rfind("/prefill", 0) == 0) {
      current_prefill_layers = parseLayers(user_input);
      std::cout << "Prefill layers: " << nlohmann::json(current_prefill_layers).dump() << "\n";
      continue;
    }

    nlohmann::json user_msg;
    user_msg["role"] = "user";
    user_msg["content"] = user_input;
    history.push_back(user_msg);

    nlohmann::json req;
    req["model"] = "mllmTeam/Qwen3-Probing";
    req["messages"] = history;
    req["prefill_layers"] = current_prefill_layers;
    req["enable_thinking"] = false;
    req["id"] = "chat-probing";

    mllm::service::sendRequest(req.dump());

    std::string assistant_content;
    bool thinking = false;

    while (true) {
      std::string resp = mllm::service::getResponse("chat-probing");
      auto j = nlohmann::json::parse(resp);

      if (j.contains("choices") && j["choices"].size() > 0) {
        auto& choice = j["choices"][0];
        auto content = choice["delta"]["content"];

        if (content.is_string()) {
          std::string s = content.get<std::string>();
          if (s.find("early_exit") != std::string::npos) {
            try {
              auto warn = nlohmann::json::parse(s);
              fmt::print(fmt::fg(fmt::color::red) | fmt::emphasis::bold,
                         "\n[Hallucination] Phase: {} | Layer: {} | Score: {:.4f}\n", warn.value("phase", "unknown"),
                         warn.value("layer", -1), warn.value("score", 0.0f));
            } catch (...) { fmt::print(fmt::fg(fmt::color::red), "\n[Hallucination] Raw: {}\n", s); }

            if (!history.empty() && history.back()["role"] == "user") history.pop_back();
            break;
          }

          if (s == "<think>") {
            thinking = true;
            continue;
          }
          if (s == "</think>") {
            thinking = false;
            continue;
          }

          if (thinking)
            fmt::print(fmt::fg(fmt::color::gray), "{}", s);
          else {
            fmt::print("{}", s);
            assistant_content += s;
          }
          std::fflush(stdout);
        }

        if (choice["finish_reason"] == "stop") break;
      }
    }

    if (!assistant_content.empty()) {
      nlohmann::json msg;
      msg["role"] = "assistant";
      msg["content"] = assistant_content;
      history.push_back(msg);
    }
  }
  mllm::service::stopService();
  return 0;
})