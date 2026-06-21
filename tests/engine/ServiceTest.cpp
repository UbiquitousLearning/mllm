#include "mllm/mllm.hpp"
#include "mllm/engine/service/Service.hpp"

MLLM_MAIN({
  mllm::service::Service::instance().start();

  // send a request
  mllm::service::RequestPayload req;
  req["model"] = "<none>";
  req["id"] = "chat-01";
  mllm::service::sendRequest(req.dump());

  // worker has already run in another thread

  // get response
  while (true) {
    std::string resp = mllm::service::getResponse("chat-01");
    auto j = nlohmann::json::parse(resp);
    std::cout << j["data"].get<std::string>() << std::flush;
    if (j["finished"]) break;
  }
  mllm::service::Service::instance().stop();
});
