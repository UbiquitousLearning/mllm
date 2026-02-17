// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "QNNUtils.hpp"

namespace mllm::qnn {

using ModelError_t = enum ModelError {
  MODEL_NO_ERROR = 0,
  MODEL_TENSOR_ERROR = 1,
  MODEL_PARAMS_ERROR = 2,
  MODEL_NODES_ERROR = 3,
  MODEL_GRAPH_ERROR = 4,
  MODEL_CONTEXT_ERROR = 5,
  MODEL_GENERATION_ERROR = 6,
  MODEL_SETUP_ERROR = 7,
  MODEL_INVALID_ARGUMENT_ERROR = 8,
  MODEL_FILE_ERROR = 9,
  MODEL_MEMORY_ALLOCATE_ERROR = 10,
  // Value selected to ensure 32 bits.
  MODEL_UNKNOWN_ERROR = 0x7FFFFFFF
};

class QNNModel {
 public:
  QNNModel(QNN_INTERFACE_VER_TYPE& qnnInterface, Qnn_BackendHandle_t backendHandle)
      : qnnInterface_(qnnInterface), backendHandle_(backendHandle) {
    if (backendHandle == nullptr) { MLLM_ERROR("QNNModel::initialize() nullptr passed as backend handle."); }
  }
  ~QNNModel() = default;

  ModelError_t initialize(const Qnn_ContextHandle_t& context, const char* graphName, bool debug, uint8_t doNodeValidations = 1,
                          const QnnGraph_Config_t** graphConfigs = nullptr);

  // Initialize from existing context and retrieved graph
  ModelError_t initializeFromContext(const Qnn_ContextHandle_t& context, const char* graphName, Qnn_GraphHandle_t graph,
                                     const Qnn_Tensor_t* inputTensors = nullptr, uint32_t numInputTensors = 0,
                                     const Qnn_Tensor_t* outputTensors = nullptr, uint32_t numOutputTensors = 0);

  // Add tensor wrapper to the model
  ModelError_t addTensorWrapper(const std::shared_ptr<QNNTensorWrapper>& tensorWrapper);

  // Add tensor using mllm::Tensor
  ModelError_t addTensor(const std::string& tensorName, Qnn_TensorType_t type, const Tensor& tensor,
                         Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);

  // Add static tensor using mllm::Tensor
  ModelError_t addStaticTensor(const std::string& tensorName, const Tensor& tensor,
                               Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);

  // Get tensor wrapper by name
  std::shared_ptr<QNNTensorWrapper> getTensorWrapper(const std::string& tensorName);

  // Add node using wrapper-based parameters
  ModelError_t addNode(Qnn_OpConfigVersion_t version, const std::string& name, const std::string& packageName,
                       const std::string& type, const std::vector<std::shared_ptr<QNNParamTensorWrapper>>& tensorParams,
                       const std::vector<std::shared_ptr<QNNParamScalarWrapper>>& scalarParams,
                       const std::vector<std::string>& inputNames, const std::vector<std::string>& outputNames);

  ModelError_t finalizeGraph(Qnn_ProfileHandle_t profileHandle, Qnn_SignalHandle_t signalHandle);

  Qnn_GraphHandle_t getQnnGraph() { return graph_; }

  std::string getQnnGraphName() { return graphName_; }

  // Get input/output tensor wrappers
  std::vector<std::shared_ptr<QNNTensorWrapper>> getGraphInputTensorWrappers() { return inputTensorWrappers_; }
  std::vector<std::shared_ptr<QNNTensorWrapper>> getGraphOutputTensorWrappers() { return outputTensorWrappers_; }

  std::map<std::string, std::vector<std::string>> getOutputTensorMap() { return modelOutputTensorMap_; }

  // Load input/output tensor information from existing graph
  ModelError_t loadGraphTensorInfo(const Qnn_Tensor_t* inputTensors, uint32_t numInputTensors,
                                   const Qnn_Tensor_t* outputTensors, uint32_t numOutputTensors);
  ModelError_t loadGraphTensorInfo();

  ModelError_t freeCachedTensors();

  [[nodiscard]] bool isGraphFinalized() const { return isFinalized_; }

 private:
  Qnn_GraphHandle_t graph_ = nullptr;
  std::string graphName_;
  bool debug_ = false;  // flag to indicate if requested graph is to be run in debug mode
  // (i.e. all intermediate tensors will be accessible to client)
  // flag to indicate whether all addNode calls need to be validated
  bool doNodeValidations_ = true;
  bool isFinalized_ = false;

  // Wrapper-based resource management
  std::vector<std::shared_ptr<QNNTensorWrapper>> inputTensorWrappers_;
  std::vector<std::shared_ptr<QNNTensorWrapper>> outputTensorWrappers_;
  std::map<std::string, std::shared_ptr<QNNTensorWrapper>> tensorWrapperMap_;
  std::vector<std::shared_ptr<QNNParamTensorWrapper>> paramTensorWrappers_;
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> paramScalarWrappers_;

  std::map<std::string, std::vector<std::string>> modelOutputTensorMap_;

  // Storage for node string parameters to ensure lifetime
  struct NodeStringStorage {
    std::string name;
    std::string packageName;
    std::string type;
  };
  std::vector<NodeStringStorage> nodeStringStorage_;

  // Qnn Backend Interface Api
  QNN_INTERFACE_VER_TYPE& qnnInterface_;
  Qnn_BackendHandle_t backendHandle_;

};  // QNN_MODEL_CLASS

// A helper function to convert QnnModel objects to Graph struct for qnn_model c
ModelError_t getGraphInfoFromModel(QNNModel& model, GraphInfoPtr_t* graphInfoPtr);

}  // namespace mllm::qnn
