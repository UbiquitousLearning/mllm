/**
 * @file mllm.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/mllm.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm {

void shutdownContext() {
  // TODO
}

void setRandomSeed(uint64_t seed) { Context::instance().setRandomSeed(seed); }

void setMaximumNumThreads(uint32_t num_threads) {
  // TODO
}

void memoryReport() { Context::instance().memoryManager()->report(); }

bool isOpenCLAvailable() {
  // TODO
  return false;
}

bool isQnnAvailable() {
  // TODO
  return false;
}

SessionTCB::ptr_t thisThread() { return Context::instance().thisThread(); }

ParameterFile::ptr_t load(const std::string& file_name, ModelFileVersion v, DeviceTypes map_2_device) {
  if (v == ModelFileVersion::kV1 && map_2_device == kCPU) {
    return ParameterFileIOImpl<kCPU, ModelFileVersion::kV1>::read(file_name);
  }

  // return empty if not match all.
  return ParameterFile::create();
}

}  // namespace mllm
