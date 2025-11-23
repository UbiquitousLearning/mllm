// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#ifndef MLLM_C_API_RUNTIME_H_
#define MLLM_C_API_RUNTIME_H_

#include <stddef.h>  // NOLINT
#include "mllm/c_api/Object.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Mllm main function
//===----------------------------------------------------------------------===//
MllmCAny initializeContext();

MllmCAny shutdownContext();

MllmCAny memoryReport();

int32_t isOk(MllmCAny ret);

//===----------------------------------------------------------------------===//
// Mllm wrapper functions
//===----------------------------------------------------------------------===//
MllmCAny convert2String(char* ptr, size_t size);

MllmCAny convert2ByteArray(char* ptr, size_t size);

MllmCAny convert2Int(int64_t v);

MllmCAny convert2Float(double v);
//===----------------------------------------------------------------------===//
// Mllm service functions
//===----------------------------------------------------------------------===//
MllmCAny startService(size_t worker_threads);

MllmCAny stopService();

void setLogLevel(int level);

MllmCAny createQwen3Session(const char* model_path);

MllmCAny createDeepseekOCRSession(const char* model_path);

MllmCAny insertSession(const char* session_id, MllmCAny handle);

MllmCAny freeSession(MllmCAny handle);

MllmCAny sendRequest(const char* session_id, const char* json_request);

const char* pollResponse(const char* session_id);

void freeResponseString(const char* response_str);

#ifdef __cplusplus
}
#endif
#endif  //! MLLM_C_API_RUNTIME_H_
