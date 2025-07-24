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

namespace mllm {

void shutdownContext() {
  // TODO
}

void setRandomSeed(uint32_t seed) {
  // TODO
}

void setMaximumNumThreads(uint32_t num_threads) {
  // TODO
}

void memoryReport() {
  // TODO
}

bool isOpenCLAvailable() {
  // TODO
  return false;
}

bool isQnnAvailable() {
  // TODO
  return false;
}

SessionTCB::ptr_t thisThread() { return Context::instance().thisThread(); }

ParameterFile::ptr_t load(const std::string& file_name, DeviceTypes map_2_device) {
  // Judge the version of model file

  // load model file

  return ParameterFile::create();
}

}  // namespace mllm
