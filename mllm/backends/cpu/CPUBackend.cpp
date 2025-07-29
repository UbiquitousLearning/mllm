/**
 * @file CPUBackend.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#include "mllm/backends/cpu/CPUBackend.hpp"
#include "mllm/backends/cpu/CPUAllocator.hpp"

// Ops
#include "mllm/backends/cpu/ops/FillOp.hpp"
#include "mllm/backends/cpu/ops/GraphOps.hpp"
#include "mllm/backends/cpu/ops/LinearOp.hpp"

namespace mllm::cpu {

CPUBackend::CPUBackend() : Backend(kCPU, createCPUAllocator()) {
  regOpFactory<CPULinearOpFactory, CPUFillOpFactory, CPUGraphBeginOpFactory, CPUGraphEndOpFactory>();
}

std::shared_ptr<CPUBackend> createCPUBackend() { return std::make_shared<CPUBackend>(); }

}  // namespace mllm::cpu
