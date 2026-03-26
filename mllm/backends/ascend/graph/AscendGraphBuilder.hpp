// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <atb/atb_infer.h>
#include <memory>
#include <string>
#include <vector>

namespace mllm::ascend {

/**
 * @brief Wrapper around ATB GraphOpBuilder for constructing computational graphs
 *
 * This class provides a high-level interface to ATB's GraphOpBuilder using the
 * TensorName API (recommended by Huawei documentation). It simplifies graph
 * construction by managing the builder lifecycle and providing clear APIs.
 *
 * Usage:
 *   AscendGraphBuilder builder;
 *   builder.beginGraph("MyGraph", {"input"}, {"output"});
 *   builder.addOperation(op1, {"input"}, {"intermediate"});
 *   builder.addOperation(op2, {"intermediate"}, {"output"});
 *   atb::Operation* graph = builder.build();
 */
class AscendGraphBuilder {
 public:
  AscendGraphBuilder();
  ~AscendGraphBuilder();

  /**
   * @brief Initialize a new graph with input/output tensor names
   *
   * @param graph_name Name of the graph (for debugging/profiling)
   * @param input_names Names of input tensors (e.g., {"hidden_states", "sin_emb"})
   * @param output_names Names of output tensors (e.g., {"output"})
   */
  void beginGraph(const std::string& graph_name,
                  const std::vector<std::string>& input_names,
                  const std::vector<std::string>& output_names,
                  atb::InferShapeFunc infer_shape_func = nullptr);

  /**
   * @brief Add an operation to the graph
   *
   * @param op Pre-created ATB operation (created via atb::CreateOperation)
   * @param input_names Names of input tensors for this operation
   * @param output_names Names of output tensors from this operation
   *
   * Note: The operation object must remain valid until build() is called.
   * Operations should be created once and can be reused across multiple graphs.
   */
  void addOperation(atb::Operation* op,
                    const std::vector<std::string>& input_names,
                    const std::vector<std::string>& output_names);

  /**
   * @brief Build and return the final graph operation
   *
   * @return Pointer to the constructed graph operation
   *         Caller is responsible for calling atb::DestroyOperation()
   *
   * Note: After calling build(), this builder can be reused for a new graph
   * by calling beginGraph() again.
   */
  atb::Operation* build();

  /**
   * @brief Get the current graph name
   */
  const std::string& graphName() const { return current_graph_name_; }

 private:
  atb::GraphOpBuilder* builder_;
  std::string current_graph_name_;

  /**
   * @brief Create a generic shape inference function
   *
   * This function is passed to GraphOpBuilder::Init(). For most cases,
   * ATB can infer shapes automatically. This provides a simple fallback.
   */
  static atb::InferShapeFunc createInferShapeFunc();

  // Disable copy
  AscendGraphBuilder(const AscendGraphBuilder&) = delete;
  AscendGraphBuilder& operator=(const AscendGraphBuilder&) = delete;
};

}  // namespace mllm::ascend
