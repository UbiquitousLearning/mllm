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

#ifdef __cplusplus
}
#endif
#endif  //! MLLM_C_API_RUNTIME_H_
