/**
 * @file Pattern.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-30
 *
 */
#include "mllm/compile/passes/Pattern.hpp"

namespace mllm::ir {

void Pattern::setIRContext(IRContext* ctx) { ir_ctx_ = ctx; }

}  // namespace mllm::ir
