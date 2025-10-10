// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <nlohmann/json.hpp>

#include "mllm/engine/service/Session.hpp"

// The service we provide is on infra side. The network communication is done by the high-level server. Maybe a golang binding
// of mllm or python binding of mllm. Golang and Python as a high-level server interface is recommended, because they are easier
// to use and has many packages.

namespace mllm::service {

using Response = std::string;
using Request = std::string;
using ResponsePayload = nlohmann::json;
using RequestPayload = nlohmann::json;

struct RequestItem {
  std::string id;
  RequestPayload payload;
  std::chrono::steady_clock::time_point enqueue_time;
};

struct ResponseItem {
  bool finished = false;
  std::string id;
  Response raw;
  ResponsePayload payload;
};

class RequestPool {
 public:
  void push(RequestItem item);

  std::optional<RequestItem> pop();

  void shutdown();

 private:
  std::queue<RequestItem> queue_;
  std::mutex mtx_;
  std::condition_variable cv_;
  bool stop_ = false;
};

class SessionPool {
 public:
  std::shared_ptr<Session> get(const std::string& session_id);

  void registerSession(const std::string& session_id, const std::shared_ptr<Session>& session);

 private:
  std::unordered_map<std::string, std::shared_ptr<Session>> sessions_;
  std::mutex mtx_;
};

class ResponsePool {
 public:
  void push(const std::string& req_id, ResponseItem item);

  std::optional<ResponseItem> pop(const std::string& req_id);

  void shutdown();

 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::unordered_map<std::string, std::queue<ResponseItem>> queues_;
};

// Singleton,
//
// Service include RequestPool, SessionPool, ResponsePool
class Service {
 public:
  static inline Service& instance() {
    static Service ins;
    return ins;
  }

  void start(size_t worker_threads = 1);

  void stop();

  RequestPool& requestPool();

  ResponsePool& responsePool();

  SessionPool& sessionPool();

 private:
  Service() = default;

  void workerLoop();

  std::atomic<bool> running_{false};
  RequestPool req_pool_;
  ResponsePool resp_pool_;
  SessionPool sess_pool_;
  std::vector<std::thread> workers_;
};

// Golang, Python SDK should bind those 5 function and provide high level Network API.
// We just focus on the server side.
void startService(size_t worker_threads = 1);

void stopService();

void insertSession(const std::string& session_id, const std::shared_ptr<Session>& session);

int sendRequest(const std::string& json_str);

Response getResponse(const std::string& id);

}  // namespace mllm::service
