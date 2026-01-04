// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/aot/passes/PTQPass.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/compile/ir/cf/Op.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::qnn::aot {

namespace {

void solveStaticWeights() {}

void solveStaticRoPE() {}

}  // namespace

uint8_t PTQPass::run(const ir::node_ptr_t& op) { return ir::PASS_RET_SUCCESS; }

ir::Pass::ptr_t createPTQPass() { return std::make_shared<PTQPass>(); }

}  // namespace mllm::qnn::aot
