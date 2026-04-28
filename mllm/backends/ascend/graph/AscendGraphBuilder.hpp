// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <atb/atb_infer.h>
#include <memory>
#include <string>
#include <vector>

namespace mllm::ascend {

// Wrapper around ATB GraphOpBuilder for constructing computational graphs.
class AscendGraphBuilder {
 public:
  // Construct an empty graph builder.
  AscendGraphBuilder();

  // Release graph builder resources.
  ~AscendGraphBuilder();

  // Initialize a new graph with input/output tensor names.
  void beginGraph(const std::string& graph_name,
                  const std::vector<std::string>& input_names,
                  const std::vector<std::string>& output_names,
                  atb::InferShapeFunc infer_shape_func = nullptr);

  // Add an operation to the graph.
  void addOperation(atb::Operation* op,
                    const std::vector<std::string>& input_names,
                    const std::vector<std::string>& output_names);

  // Create a reshaped tensor view inside the graph.
  void reshape(const std::string& src_tensor_name,
               atb::ReshapeFunc reshape_func,
               const std::string& view_tensor_name);

  // Build and return the final graph operation.
  atb::Operation* build();

  // Get the current graph name.
  const std::string& graphName() const { return current_graph_name_; }

 private:
  atb::GraphOpBuilder* builder_;
  std::string current_graph_name_;

  // Create a generic shape inference function.
  static atb::InferShapeFunc createInferShapeFunc();

  // Disable copy construction.
  AscendGraphBuilder(const AscendGraphBuilder&) = delete;

  // Disable copy assignment.
  AscendGraphBuilder& operator=(const AscendGraphBuilder&) = delete;
};

}  // namespace mllm::ascend
