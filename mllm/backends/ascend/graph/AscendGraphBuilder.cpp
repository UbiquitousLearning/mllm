// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "AscendGraphBuilder.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/utils/Common.hpp"
#include <iostream>

namespace mllm::ascend {

AscendGraphBuilder::AscendGraphBuilder() : builder_(nullptr), current_graph_name_("") {
  // GraphOpBuilder construction relies on initialized ACL/ATB runtime.
  (void)getGlobalAtbContext();

  auto ret = atb::CreateGraphOpBuilder(&builder_);
  if (ret != atb::NO_ERROR || builder_ == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "CreateGraphOpBuilder failed, status={}", static_cast<int>(ret));
  }
}

AscendGraphBuilder::~AscendGraphBuilder() {
  if (builder_ != nullptr) {
    atb::DestroyGraphOpBuilder(builder_);
    builder_ = nullptr;
  }
}

void AscendGraphBuilder::beginGraph(
    const std::string& graph_name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    atb::InferShapeFunc infer_shape_func) {

  current_graph_name_ = graph_name;

  // Create shape inference function
  auto inferShapeFunc = infer_shape_func ? infer_shape_func : createInferShapeFunc();

  // Convert std::vector to atb::SVector
  atb::SVector<std::string> atb_input_names;
  for (const auto& name : input_names) {
    atb_input_names.push_back(name);
  }
  atb::SVector<std::string> atb_output_names;
  for (const auto& name : output_names) {
    atb_output_names.push_back(name);
  }

  // Initialize graph with TensorName API
  auto ret = builder_->Init(
      graph_name.c_str(),
      inferShapeFunc,
      atb_input_names,
      atb_output_names
  );

  if (ret != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "GraphOpBuilder::Init failed for '{}', status={}",
                    graph_name, static_cast<int>(ret));
  }
}

void AscendGraphBuilder::addOperation(
    atb::Operation* op,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names) {

  if (op == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "Cannot add null operation to graph '{}'", current_graph_name_);
  }

  // Convert std::vector to atb::SVector
  atb::SVector<std::string> atb_input_names;
  for (const auto& name : input_names) {
    atb_input_names.push_back(name);
  }
  atb::SVector<std::string> atb_output_names;
  for (const auto& name : output_names) {
    atb_output_names.push_back(name);
  }

  auto ret = builder_->AddOperation(op, atb_input_names, atb_output_names);

  if (ret != atb::NO_ERROR) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "AddOperation failed for graph '{}', status={}",
                    current_graph_name_, static_cast<int>(ret));
  }
}

atb::Operation* AscendGraphBuilder::build() {
  atb::Operation* graphOp = builder_->Build();

  if (graphOp == nullptr) {
    MLLM_ERROR_EXIT(ExitCode::kAscendError,
                    "GraphOpBuilder::Build failed for graph '{}'",
                    current_graph_name_);
  }

  return graphOp;
}

atb::InferShapeFunc AscendGraphBuilder::createInferShapeFunc() {
  // Generic shape inference function
  // For most ATB operators, ATB can infer shapes automatically
  // This provides a simple fallback that copies the first input shape to first output
  return [](const atb::SVector<atb::TensorDesc>& inTensorDescs,
            atb::SVector<atb::TensorDesc>& outTensorDescs) -> atb::Status {
    // Simple heuristic: output shape matches first input shape
    // ATB's internal shape inference will override this for most operators
    if (!inTensorDescs.empty() && !outTensorDescs.empty()) {
      outTensorDescs.at(0) = inTensorDescs.at(0);
    }
    return atb::NO_ERROR;
  };
}

}  // namespace mllm::ascend
