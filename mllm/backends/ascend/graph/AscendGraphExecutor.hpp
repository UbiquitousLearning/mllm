// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <atb/atb_infer.h>
#include <cstdint>
#include <vector>
#include "mllm/core/Tensor.hpp"

namespace mllm::ascend {

/**
 * @brief Executor for ATB graph operations with persistent workspace management
 *
 * This class manages the execution of ATB graph operations, handling workspace
 * allocation and reuse across multiple executions. It follows RAII principles
 * to ensure proper cleanup of resources.
 *
 * Key features:
 * - Persistent workspace: Allocate once, reuse across executions
 * - Automatic workspace resizing if needed
 * - Stream synchronization after execution
 * - RAII cleanup of graph operation and workspace
 *
 * Usage:
 *   atb::Operation* graph = builder.build();
 *   AscendGraphExecutor executor(graph, context);
 *   executor.execute({input_tensor}, {output_tensor});
 *   // Executor destructor will clean up graph and workspace
 */
class AscendGraphExecutor {
 public:
  /**
   * @brief Construct executor for a graph operation
   *
   * @param graph_op Pointer to the graph operation (ownership transferred)
   * @param context ATB context for execution
   *
   * Note: The executor takes ownership of graph_op and will call
   * atb::DestroyOperation() in the destructor.
   */
  AscendGraphExecutor(atb::Operation* graph_op, atb::Context* context);

  ~AscendGraphExecutor();

  /**
   * @brief Execute the graph with given inputs and outputs
   *
   * @param inputs Vector of input tensors (must match graph input count/types)
   * @param outputs Vector of output tensors (must match graph output count/types)
   *
   * The method:
   * 1. Converts MLLM tensors to ATB tensors
   * 2. Calls graph->Setup() to determine workspace size
   * 3. Allocates/resizes workspace if needed
   * 4. Executes the graph
   * 5. Synchronizes the stream
   */
  void execute(const std::vector<Tensor>& inputs,
               std::vector<Tensor>& outputs);

  /**
   * @brief Get current workspace size in bytes
   */
  uint64_t workspaceSize() const { return workspace_size_; }

  /**
   * @brief Get the graph operation pointer
   */
  atb::Operation* graphOp() const { return graph_op_; }

 private:
  atb::Operation* graph_op_;
  atb::Context* context_;
  void* workspace_;
  uint64_t workspace_size_;

  // Disable copy and move
  AscendGraphExecutor(const AscendGraphExecutor&) = delete;
  AscendGraphExecutor& operator=(const AscendGraphExecutor&) = delete;
  AscendGraphExecutor(AscendGraphExecutor&&) = delete;
  AscendGraphExecutor& operator=(AscendGraphExecutor&&) = delete;
};

}  // namespace mllm::ascend
