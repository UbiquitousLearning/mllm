/**
 * @file x86Backend.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#include "mllm/backends/x86/X86Backend.hpp"
#include "mllm/backends/x86/X86Allocator.hpp"

// Ops
#include "mllm/backends/x86/ops/LinearOp.hpp"

namespace mllm::x86 {

X86Backend::X86Backend() : Backend(kCPU, createX86Allocator()) { regOpFactory<X86LinearOpFactory>(); }

std::shared_ptr<X86Backend> createX86Backend() { return std::make_shared<X86Backend>(); }

}  // namespace mllm::x86
