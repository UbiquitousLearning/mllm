#include <iostream>
#include <mllm/mllm.hpp>
#include <mllm/engine/service/Service.hpp>
#include <mllm/models/qwen3/modeling_qwen3_service.hpp>

MLLM_MAIN({
  auto& model_path = mllm::Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  mllm::Argparse::parse(argc, argv);

  auto qwen3_session = std::make_shared<mllm::models::qwen3::Qwen3Session>();
  qwen3_session->fromPreTrain(model_path.get());
  mllm::service::insertSession("mllmTeam/Qwen3-0.6B-w4a32kai", qwen3_session);
  mllm::service::startService();

  // Build Request
  mllm::service::RequestPayload req;
  req["model"] = "mllmTeam/Qwen3-0.6B-w4a32kai";
  mllm::service::RequestPayload one_msg;
  one_msg["role"] = "user";
  one_msg["content"] = "Say Hello in Chinese, English, French and German.";
  req["messages"] = json::array({one_msg});
  req["id"] = "chat-01";
  req["enable_thinking"] = true;
  mllm::service::sendRequest(req.dump());

  // getResponse will block until the response is ready, while(true) is ok in this case.
  while (true) {
    std::string resp = mllm::service::getResponse("chat-01");
    auto j = nlohmann::json::parse(resp);
    std::cout << j["data"].get<std::string>() << std::flush;
    if (j["finished"]) break;
  }

  mllm::service::stopService();
})
