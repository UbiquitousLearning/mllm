// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/compile/passes/Pattern.hpp"

namespace mllm::ir {

void Pattern::setIRContext(IRContext* ctx) { ir_ctx_ = ctx; }

}  // namespace mllm::ir
