// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm {

void shutdownContext() {
  auto all_threads = Context::instance().refSessionThreads();
  auto this_thread = Context::instance().thisThread();
  for (auto& tcb : all_threads) {
    if (tcb.first != this_thread->system_tid) {
      tcb.second->attached_contexts._ref_raw_data().clear();
      tcb.second->layer_ops_table._ref_raw_data().clear();
      tcb.second->ir_context = nullptr;
      tcb.second->trace_mode = false;
    }
  }
  ::mllm::cleanThisThread();
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

void perfStart() { Context::instance().setPerfMode(true); }

void perfEnd() {
  Context::instance().setPerfMode(false);
  Context::instance().getPerfFile()->finalize();
}

void cleanThisThread() {
  Context::instance().thisThread()->attached_contexts._ref_raw_data().clear();
  Context::instance().thisThread()->layer_ops_table._ref_raw_data().clear();
  Context::instance().thisThread()->ir_context = nullptr;
  Context::instance().thisThread()->trace_mode = false;
}

PerfFile::ptr_t getPerfFile() { return Context::instance().getPerfFile(); }

SessionTCB::ptr_t thisThread() { return Context::instance().thisThread(); }

ParameterFile::ptr_t load(const std::string& file_name, ModelFileVersion v, DeviceTypes map_2_device) {
  if (v == ModelFileVersion::kV1 && map_2_device == kCPU) {
    return ParameterFileIOImpl<kCPU, ModelFileVersion::kV1>::read(file_name);
  }

  // return empty if not match all.
  return ParameterFile::create();
}

}  // namespace mllm
