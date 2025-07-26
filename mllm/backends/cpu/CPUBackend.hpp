/**
 * @file CPUBackend.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#pragma once

#include <memory>
#include "mllm/backends/base/Backend.hpp"

namespace mllm::cpu {

class CPUBackend final : public Backend {
 public:
  CPUBackend();
};

std::shared_ptr<CPUBackend> createCPUBackend();
}  // namespace mllm::cpu
