// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <memory>
#include <chrono>
#include <optional>

#include "mllm/engine/service/Session.hpp"
#include "mllm/engine/service/Service.hpp"

namespace mllm::service {

//===----------------------------------------------------------------------===//
// RequestPool
//===----------------------------------------------------------------------===//
void RequestPool::push(RequestItem item) {
  {
    std::lock_guard lk(mtx_);
    queue_.push(std::move(item));
  }
  cv_.notify_one();
}

std::optional<RequestItem> RequestPool::pop() {
  std::unique_lock lk(mtx_);
  cv_.wait(lk, [this] { return !queue_.empty() || stop_; });

  if (stop_ && queue_.empty()) { return std::nullopt; }

  auto item = std::move(queue_.front());
  queue_.pop();
  return item;
}

void RequestPool::shutdown() {
  {
    std::lock_guard lk(mtx_);
    stop_ = true;
  }
  cv_.notify_all();
}

//===----------------------------------------------------------------------===//
// ResponsePool
//===----------------------------------------------------------------------===//
void ResponsePool::push(const std::string& req_id, ResponseItem item) {
  std::unique_lock lk(mtx_);
  auto& q = queues_[req_id];
  q.push(std::move(item));
  cv_.notify_one();
}

std::optional<ResponseItem> ResponsePool::pop(const std::string& req_id) {
  std::unique_lock lk(mtx_);
  cv_.wait(lk, [this, &req_id] {
    auto it = queues_.find(req_id);
    return (it != queues_.end() && !it->second.empty()) || stop_;
  });
  if (stop_) return std::nullopt;

  auto& q = queues_[req_id];
  auto v = std::move(q.front());
  q.pop();
  if (q.empty()) queues_.erase(req_id);
  return v;
}

void ResponsePool::shutdown() {
  std::lock_guard lk(mtx_);
  stop_ = true;
  cv_.notify_all();
}

//===----------------------------------------------------------------------===//
// SessionPool
//===----------------------------------------------------------------------===//
std::shared_ptr<Session> SessionPool::get(const std::string& session_id) {
  if (sessions_.count(session_id)) { return sessions_[session_id]; }
  return nullptr;
}

void SessionPool::registerSession(const std::string& session_id, const std::shared_ptr<Session>& session) {
  if (sessions_.count(session_id)) {
    // TODO
    return;
  }
  sessions_[session_id] = session;
}

//===----------------------------------------------------------------------===//
// Service
//===----------------------------------------------------------------------===//
void Service::start(size_t worker_threads) {
  running_ = true;
  sess_pool_.registerSession("<none>", std::make_shared<NoneSession>());
  for (size_t i = 0; i < worker_threads; ++i) workers_.emplace_back([this] { workerLoop(); });
}

void Service::stop() {
  req_pool_.shutdown();
  running_ = false;

  for (auto& t : workers_) {
    if (t.joinable()) { t.join(); }
  }

  resp_pool_.shutdown();
}

RequestPool& Service::requestPool() { return req_pool_; }

ResponsePool& Service::responsePool() { return resp_pool_; }

SessionPool& Service::sessionPool() { return sess_pool_; }

void Service::workerLoop() {
  while (running_) {
    try {
      if (auto req_opt = req_pool_.pop(); req_opt) {
        RequestItem& req = *req_opt;
        auto session = sess_pool_.get(req.payload.value("model", "<none>"));

        session->streamGenerate(
            req.payload, [this, req_payload = req.payload, req_id = req.id](const std::string& token, bool finished) {
              ResponseItem item;
              item.id = req_id;
              item.finished = finished;
              item.raw = token;

              std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

              // Make payload
              item.payload = {
                  {"model", req_payload["model"]},
                  {"created", now},
                  {"choices", nlohmann::json::array({{{"index", 0},
                                                      {"delta", {{"content", token}}},
                                                      {"finish_reason", finished ? "stop" : nlohmann::json(nullptr)}}})}};

              resp_pool_.push(req_id, std::move(item));
            });
      } else {
        // pop return empty, which means service is stopped and queue is empty.
        break;
      }
    } catch (...) {
      // TODO
    }
  }
}

void startService(size_t worker_threads) { Service::instance().start(worker_threads); }

void stopService() { Service::instance().stop(); }

void insertSession(const std::string& session_id, const std::shared_ptr<Session>& session) {
  Service::instance().sessionPool().registerSession(session_id, session);
}

int sendRequest(const std::string& json_str) {
  if (json_str.empty()) return -1;
  try {
    auto j = nlohmann::json::parse(json_str);
    RequestItem item;
    item.id = j.value("id", "");
    item.payload = j;
    item.enqueue_time = std::chrono::steady_clock::now();
    Service::instance().requestPool().push(std::move(item));
    return 0;
  } catch (...) {
    // TODO
    return -1;
  }
}

Response getResponse(const std::string& id) {
  if (id.empty()) {
    nlohmann::json j;
    j["error"] = true;
    j["finished"] = true;
    return j.dump();
  }
  try {
    auto& pool = Service::instance().responsePool();
    auto opt = pool.pop(id);
    if (!opt) {
      nlohmann::json j;
      j["finished"] = false;
      return j.dump();
    }
    nlohmann::json j = opt->payload;
    j["id"] = opt->id;
    j["object"] = "chat.completion.chunk";
    j["finished"] = opt->finished;
    std::string s = j.dump();
    return s;
  } catch (...) {
    // TODO
    nlohmann::json j;
    j["error"] = true;
    j["finished"] = true;
    return j.dump();
  }
}

}  // namespace mllm::service
