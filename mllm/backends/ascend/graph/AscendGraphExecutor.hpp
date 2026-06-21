// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <atb/atb_infer.h>
#include <cstdint>
#include <vector>
#include "mllm/core/Tensor.hpp"

namespace mllm::ascend {

// Executor for ATB graph operations with persistent workspace management.
class AscendGraphExecutor {
 public:
  // Construct executor for a graph operation and take ownership of it.
  AscendGraphExecutor(atb::Operation* graph_op, atb::Context* context);

  // Release graph operation and workspace resources.
  ~AscendGraphExecutor();

  // Execute the graph with given inputs and outputs.
  void execute(const std::vector<Tensor>& inputs,
               std::vector<Tensor>& outputs);

  // Get current workspace size in bytes.
  uint64_t workspaceSize() const { return workspace_size_; }

  // Get the graph operation pointer.
  atb::Operation* graphOp() const { return graph_op_; }

 private:
  atb::Operation* graph_op_;
  atb::Context* context_;
  void* workspace_;
  uint64_t workspace_size_;
  int workspace_block_id_;

  // Disable copy construction.
  AscendGraphExecutor(const AscendGraphExecutor&) = delete;

  // Disable copy assignment.
  AscendGraphExecutor& operator=(const AscendGraphExecutor&) = delete;

  // Disable move construction.
  AscendGraphExecutor(AscendGraphExecutor&&) = delete;

  // Disable move assignment.
  AscendGraphExecutor& operator=(AscendGraphExecutor&&) = delete;
};

}  // namespace mllm::ascend
