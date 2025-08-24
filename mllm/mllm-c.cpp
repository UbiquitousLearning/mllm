// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <mllm/mllm.hpp>

#include "mllm-c.hpp"

MllmReturnCode mllm_init_context() {
  mllm::initializeContext();
  return MLLM_SUCCESS;
}

MllmReturnCode mllm_shutdown_context() {
  mllm::shutdownContext();
  return MLLM_SUCCESS;
}

MllmReturnCode mllm_show_memory_report() {
  mllm::memoryReport();
  return MLLM_SUCCESS;
}
