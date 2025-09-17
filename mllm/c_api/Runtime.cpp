// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/mllm.hpp"
#include "mllm/c_api/Runtime.h"

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
  if (ret.type_id == kRetCode && ret.v_return_code == 0) { return true; }
  return false;
}

//===----------------------------------------------------------------------===//
// Mllm wrapper functions
//===----------------------------------------------------------------------===//
MllmCAny convert2String(char* ptr, size_t size) {
  // TODO
}

MllmCAny convert2ByteArray(char* ptr, size_t size) {
  // TODO
}

MllmCAny convert2Int(int64_t v) { return MllmCAny{.type_id = kInt, .v_int64 = v}; }

MllmCAny convert2Float(double v) { return MllmCAny{.type_id = kFloat, .v_fp64 = v}; }
