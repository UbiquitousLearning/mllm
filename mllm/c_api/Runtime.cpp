// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/mllm.hpp"
#include "mllm/c_api/Runtime.h"
#include "mllm/engine/service/Service.hpp"
#include "mllm/models/qwen3/modeling_qwen3_service.hpp"
#include "mllm/models/deepseek_ocr/modeling_deepseek_ocr_service.hpp"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <string.h> // for strncpy

struct MllmSessionWrapper {
    std::shared_ptr<mllm::service::Session> session_ptr;
};

//===----------------------------------------------------------------------===//
// Mllm main function
//===----------------------------------------------------------------------===//
MllmCAny initializeContext() {
  mllm::initializeContext();
  return MllmCAny{.type_id = kRetCode, .v_return_code = 0};
}

MllmCAny shutdownContext() {
  mllm::shutdownContext();
  return MllmCAny{.type_id = kRetCode, .v_return_code = 0};
}

MllmCAny memoryReport() {
  mllm::memoryReport();
  return MllmCAny{.type_id = kRetCode, .v_return_code = 0};
}

int32_t isOk(MllmCAny ret) {
  if (ret.type_id == kRetCode && ret.v_return_code == 0)
      return true;
  if (ret.type_id == kCustomObject && ret.v_custom_ptr != nullptr)
      return true;
  return false;
}


//===----------------------------------------------------------------------===//
// Mllm wrapper functions
//===----------------------------------------------------------------------===//
MllmCAny convert2String(char* ptr, size_t size) {
  // TODO
  return MllmCAny{.type_id = kRetCode, .v_return_code = -1};
}

MllmCAny convert2ByteArray(char* ptr, size_t size) {
  // TODO
  return MllmCAny{.type_id = kRetCode, .v_return_code = -1};
}

MllmCAny convert2Int(int64_t v) { return MllmCAny{.type_id = kInt, .v_int64 = v}; }

MllmCAny convert2Float(double v) { return MllmCAny{.type_id = kFloat, .v_fp64 = v}; }

//===----------------------------------------------------------------------===//
// Mllm service functions
//===----------------------------------------------------------------------===//

MllmCAny startService(size_t worker_threads) {
    mllm::service::startService(worker_threads);
    return MllmCAny{.type_id = kRetCode, .v_return_code = 0};
}

MllmCAny stopService() {
    mllm::service::stopService();
    return MllmCAny{.type_id = kRetCode, .v_return_code = 0};
}

void setLogLevel(int level) {
    mllm::setLogLevel(static_cast<mllm::LogLevel>(level));
}

MllmCAny createQwen3Session(const char* model_path) {
    if (model_path == nullptr) {
        printf("[C++ Service] createQwen3Session error: invalid arguments.\n");
        return MllmCAny{.type_id = kRetCode, .v_return_code = -1};
    }
    try {
        auto qwen3_session = std::make_shared<mllm::models::qwen3::Qwen3Session>();
        qwen3_session->fromPreTrain(model_path);

        auto* handle = new MllmSessionWrapper();
        handle->session_ptr = qwen3_session;

        return MllmCAny{.type_id = kCustomObject, .v_custom_ptr = handle};
    } catch (const std::exception& e) {
        printf("[C++ Service] createQwen3Session exception: %s\n", e.what());
        return MllmCAny{.type_id = kRetCode, .v_return_code = -1};
    }
}

MllmCAny createDeepseekOCRSession(const char* model_path) {
    if (model_path == nullptr) {
        printf("[C++ Service] createDeepseekOCRSession error: invalid arguments.\n");
        return MllmCAny{.type_id = kRetCode, .v_return_code = -1};
    }
    try {
        auto dpsk_session = std::make_shared<mllm::models::deepseek_ocr::DeepseekOCRSession>();
        dpsk_session->fromPreTrain(model_path);

        auto* handle = new MllmSessionWrapper();
        handle->session_ptr = dpsk_session;

        return MllmCAny{.type_id = kCustomObject, .v_custom_ptr = handle};
    } catch (const std::exception& e) {
        printf("[C++ Service] createDeepseekOCRSession exception: %s\n", e.what());
        return MllmCAny{.type_id = kRetCode, .v_return_code = -1};
    }
}

MllmCAny insertSession(const char* session_id, MllmCAny handle) {
    if (session_id == nullptr || handle.type_id != kCustomObject || handle.v_custom_ptr == nullptr) {
        printf("[C++ Service] insertSession error: invalid arguments.\n");
        return MllmCAny{.type_id = kRetCode, .v_return_code = -1};
    }

    auto* session_wrapper = reinterpret_cast<MllmSessionWrapper*>(handle.v_custom_ptr);
    mllm::service::insertSession(std::string(session_id), session_wrapper->session_ptr);
    return MllmCAny{.type_id = kRetCode, .v_return_code = 0};
}

MllmCAny freeSession(MllmCAny handle) {
    if (handle.type_id != kCustomObject || handle.v_custom_ptr == nullptr) {
        printf("[C++ Service] freeSession error: invalid arguments.\n");
        return MllmCAny{.type_id = kRetCode, .v_return_code = -1};
    }

    auto* session_wrapper = reinterpret_cast<MllmSessionWrapper*>(handle.v_custom_ptr);
    delete session_wrapper;
    return MllmCAny{.type_id = kRetCode, .v_return_code = 0};
}


MllmCAny sendRequest(const char* session_id, const char* json_request) {
    if (session_id == nullptr || json_request == nullptr) {
        printf("[C++ Service] sendRequest error: invalid arguments.\n");
        return MllmCAny{.type_id = kRetCode, .v_return_code = -1};
    }
    int status = mllm::service::sendRequest(std::string(json_request));
    return MllmCAny{.type_id = kRetCode, .v_return_code = status};
}

const char* pollResponse(const char* session_id) {
    if (session_id == nullptr) {
        return nullptr;
    }

    std::string request_id = std::string(session_id);
    mllm::service::Response response = mllm::service::getResponse(request_id);

    if (response.empty()) {
        return nullptr;
    }

    bool finished = false;
    try {
        nlohmann::json j = nlohmann::json::parse(response);


        if (j.contains("choices")) {
            if (j["choices"].is_array() && !j["choices"].empty()) {
                const auto& first_choice = j["choices"][0];
                if (first_choice.contains("finish_reason") && first_choice["finish_reason"] == "stop") {
                    finished = true;
                }
            }
        }

    } catch (const nlohmann::json::parse_error& e) {
        printf("[C++ Service] pollResponse JSON parse error: %s\n", e.what());
        return nullptr;
    }

    if (finished) {
        return nullptr; 
    }

    char* c_response = new char[response.length() + 1];
    strncpy(c_response, response.c_str(), response.length() + 1);
    
    return c_response;
}

void freeResponseString(const char* response_str) {
    if (response_str != nullptr) {
        delete[] response_str;
    }
}