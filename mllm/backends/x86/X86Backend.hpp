/**
 * @file x86Backend.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#pragma once

#include <memory>
#include "mllm/backends/base/Backend.hpp"

namespace mllm::x86 {

class X86Backend final : public Backend {
 public:
  X86Backend();
};

std::shared_ptr<X86Backend> createX86Backend();
}  // namespace mllm::x86
