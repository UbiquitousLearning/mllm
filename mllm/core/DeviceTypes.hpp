// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>

namespace mllm {

enum DeviceTypes : int32_t {
  kDeviceTypes_Start = 0,

  kCPU = 1,
  kCUDA = 2,
  kOpenCL = 3,
  kQNN = 4,

  // TODO This device type for pinned memory.
  kCUDAHost = 5,

  kDynamicDeviceType_Start = 4096,

  kDeviceTypes_End,
};

inline const char* deviceTypes2Str(DeviceTypes type) {
  switch (type) {
    case DeviceTypes::kDeviceTypes_Start: return "DeviceTypes_Start";
    case DeviceTypes::kCPU: return "CPU";
    case DeviceTypes::kCUDA: return "CUDA";
    case DeviceTypes::kOpenCL: return "OpenCL";
    case DeviceTypes::kQNN: return "QNN";
    case DeviceTypes::kDeviceTypes_End: return "DeviceTypes_End";
    default: return "Unknown";
  }
}

inline DeviceTypes str2DeviceType(const std::string& type_str) {
  if (type_str == "CPU") {
    return DeviceTypes::kCPU;
  } else if (type_str == "CUDA") {
    return DeviceTypes::kCUDA;
  } else if (type_str == "OpenCL") {
    return DeviceTypes::kOpenCL;
  } else if (type_str == "QNN") {
    return DeviceTypes::kQNN;
  } else {
    return DeviceTypes::kDeviceTypes_End;
  }
}

}  // namespace mllm

// WARN: The Macros below should not be used anymore.
//
// We left it here just for compatibility. It need to be removed one day.
#define MLLM_CPU ::mllm::DeviceTypes::kCPU
#define MLLM_OPENCL ::mllm::DeviceTypes::kOpenCL
#define MLLM_CUDA ::mllm::DeviceTypes::kCUDA
#define MLLM_QNN ::mllm::DeviceTypes::kQNN
