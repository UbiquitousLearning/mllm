// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <mllm/mllm.hpp>

#include "mllm-c.hpp"

int mllm_init_context() {
  mllm::initializeContext();
  return 0;
}

int mllm_shutdown_context() {
  mllm::shutdownContext();
  return 0;
}

int mllm_show_memory_report() {
  mllm::memoryReport();
  return 0;
}
