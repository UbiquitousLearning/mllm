// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/compile/passes/Pass.hpp"
#include "mllm/compile/ir/Node.hpp"

namespace mllm::qnn::aot {

//===----------------------------------------------------------------------===//
// PTQPass - Post-Training Quantization Pass
// This pass applies post-training quantization transformations to the IR.
// It walks through the computation graph and applies quantization
// based on configuration parameters.
//===----------------------------------------------------------------------===//
class PTQPass final : public ir::Pass {
 public:
  PTQPass() = default;

  ~PTQPass() override = default;

  // Run the PTQ pass on the given operation
  // Expected input: ModuleOp containing the computation graph
  // Output: Modified IR with PTQ transformations applied
  uint8_t run(const ir::node_ptr_t& op) override;
};

// Factory function to create PTQPass instance
ir::Pass::ptr_t createPTQPass();

}  // namespace mllm::qnn::aot
