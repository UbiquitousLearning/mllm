// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/QNNModel.hpp"
#include <cassert>
#include "mllm/backends/qnn/QNNTypeMacros.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn {

template<typename... Args>
void freeMultiPtr(Args... args) {
  (free(args), ...);
}

char* strnDup(const char* source, size_t maxlen) { return ::strndup(source, maxlen); }

ModelError_t QNNModel::initialize(const Qnn_ContextHandle_t& context, const char* graphName, bool debug,
                                  uint8_t doNodeValidations, const QnnGraph_Config_t** graphConfigs) {
  if (context == nullptr) {
    MLLM_ERROR("QNNModel::initialize() nullptr passed as context handle.");
    return MODEL_CONTEXT_ERROR;
  }
  if (graphName == nullptr) {
    MLLM_ERROR("QNNModel::initialize() nullptr passed as graphName.");
    return MODEL_GRAPH_ERROR;
  }

  if (!graphName_.empty()) {
    // only one graph is allowed per QnnModel
    MLLM_ERROR("QNNModel::initialize() model for graph {} already initialized.", graphName);
    return MODEL_GRAPH_ERROR;
  }

  if (!doNodeValidations_) {
    MLLM_WARN("Node validation disabled. Backend will not perform op validation prior to adding Node.");
  }

  graphName_ = graphName;
  debug_ = debug;
  doNodeValidations_ = doNodeValidations;

  if (qnnInterface_.graphCreate(context, graphName, graphConfigs, &graph_) != QNN_GRAPH_NO_ERROR || graph_ == nullptr) {
    MLLM_ERROR("QNNModel::initialize() not able to create graph in given context.");
    return MODEL_GRAPH_ERROR;
  }

  return MODEL_NO_ERROR;
}

ModelError_t QNNModel::initializeFromContext(const Qnn_ContextHandle_t& context, const char* graphName, Qnn_GraphHandle_t graph,
                                             const Qnn_Tensor_t* inputTensors, uint32_t numInputTensors,
                                             const Qnn_Tensor_t* outputTensors, uint32_t numOutputTensors) {
  if (context == nullptr) {
    MLLM_ERROR("QNNModel::initializeFromContext() nullptr passed as context handle.");
    return MODEL_CONTEXT_ERROR;
  }
  if (graphName == nullptr) {
    MLLM_ERROR("QNNModel::initializeFromContext() nullptr passed as graphName.");
    return MODEL_GRAPH_ERROR;
  }
  if (graph == nullptr) {
    MLLM_ERROR("QNNModel::initializeFromContext() nullptr passed as graph handle.");
    return MODEL_GRAPH_ERROR;
  }

  if (!graphName_.empty()) {
    MLLM_ERROR("QNNModel::initializeFromContext() model for graph {} already initialized.", graphName);
    return MODEL_GRAPH_ERROR;
  }

  graphName_ = graphName;
  graph_ = graph;

  // Load tensor information from the provided tensor arrays
  ModelError_t err = loadGraphTensorInfo(inputTensors, numInputTensors, outputTensors, numOutputTensors);
  if (err != MODEL_NO_ERROR) {
    MLLM_ERROR("QNNModel::initializeFromContext() failed to load graph tensor info for graph: {}", graphName);
    return err;
  }

  isFinalized_ = true;

  return MODEL_NO_ERROR;
}

ModelError_t QNNModel::loadGraphTensorInfo(const Qnn_Tensor_t* inputTensors, uint32_t numInputTensors,
                                           const Qnn_Tensor_t* outputTensors, uint32_t numOutputTensors) {
  if (graph_ == nullptr) {
    MLLM_ERROR("QNNModel::loadGraphTensorInfo() graph handle is null.");
    return MODEL_GRAPH_ERROR;
  }

  // Create wrappers for input tensors
  for (uint32_t i = 0; i < numInputTensors; ++i) {
    const Qnn_Tensor_t* tensor = &inputTensors[i];
    std::string tensorName = QNN_TENSOR_GET_NAME(tensor);

    // Create dimension vector
    std::vector<uint32_t> dimensions;
    uint32_t rank = QNN_TENSOR_GET_RANK(tensor);
    dimensions.reserve(rank);
    for (uint32_t j = 0; j < rank; ++j) { dimensions.push_back(QNN_TENSOR_GET_DIMENSIONS(tensor)[j]); }

    auto wrapper = std::make_shared<QNNTensorWrapper>(tensorName,
                                                      QNN_TENSOR_TYPE_APP_WRITE,  // Input tensors are APP_WRITE
                                                      QNN_TENSOR_GET_DATA_TYPE(tensor), dimensions, DEFAULT_QUANTIZE_PARAMS);

    // Set the native tensor to point to the retrieved tensor
    wrapper->initFromQnnTensor(const_cast<Qnn_Tensor_t*>(tensor));

    inputTensorWrappers_.push_back(wrapper);
    tensorWrapperMap_[tensorName] = wrapper;
  }

  // Create wrappers for output tensors
  for (uint32_t i = 0; i < numOutputTensors; ++i) {
    const Qnn_Tensor_t* tensor = &outputTensors[i];
    std::string tensorName = QNN_TENSOR_GET_NAME(tensor);

    // Create dimension vector
    std::vector<uint32_t> dimensions;
    uint32_t rank = QNN_TENSOR_GET_RANK(tensor);
    dimensions.reserve(rank);
    for (uint32_t j = 0; j < rank; ++j) { dimensions.push_back(QNN_TENSOR_GET_DIMENSIONS(tensor)[j]); }

    auto wrapper = std::make_shared<QNNTensorWrapper>(tensorName,
                                                      QNN_TENSOR_TYPE_APP_READ,  // Output tensors are APP_READ
                                                      QNN_TENSOR_GET_DATA_TYPE(tensor), dimensions, DEFAULT_QUANTIZE_PARAMS);

    // Set the native tensor to point to the retrieved tensor
    wrapper->initFromQnnTensor(const_cast<Qnn_Tensor_t*>(tensor));

    outputTensorWrappers_.push_back(wrapper);
    tensorWrapperMap_[tensorName] = wrapper;
    // Record QNN output order (index in outputTensorWrappers_)
    qnnOutputNameToIndex_[tensorName] = static_cast<int>(outputTensorWrappers_.size() - 1);
  }

  MLLM_INFO("QNNModel::loadGraphTensorInfo() loaded {} input tensors and {} output tensors for graph: {}", numInputTensors,
            numOutputTensors, graphName_);

  return MODEL_NO_ERROR;
}

ModelError_t QNNModel::loadGraphTensorInfo() {
  // This method can be used for future implementation if needed
  MLLM_ERROR("QNNModel::loadGraphTensorInfo() without parameters is not implemented.");
  return MODEL_NO_ERROR;
}

ModelError_t QNNModel::addTensorWrapper(const std::shared_ptr<QNNTensorWrapper>& tensorWrapper) {
  if (!tensorWrapper) {
    MLLM_ERROR("QNNModel::addTensorWrapper() NULL tensor wrapper provided.");
    return MODEL_TENSOR_ERROR;
  }

  auto nativeTensor = tensorWrapper->getNativeTensor();
  std::string tensorName = QNN_TENSOR_GET_NAME(nativeTensor);

  // Verify tensor being added is not a duplicate
  if (tensorWrapperMap_.find(tensorName) != tensorWrapperMap_.end()) {
    MLLM_ERROR("QNNModel::addTensorWrapper() tensor {} already exists.", tensorName);
    return MODEL_TENSOR_ERROR;
  }

  if (debug_ && QNN_TENSOR_GET_TYPE(nativeTensor) == QNN_TENSOR_TYPE_NATIVE) {
    // for debug, make all tensors accessible by client
    QNN_TENSOR_SET_TYPE(nativeTensor, QNN_TENSOR_TYPE_APP_READ);
  }

  if (qnnInterface_.tensorCreateGraphTensor(graph_, nativeTensor) != QNN_TENSOR_NO_ERROR) {
    MLLM_ERROR("QNNModel::addTensorWrapper() error creating tensor {}", tensorName);
    return MODEL_TENSOR_ERROR;
  }

  // Store wrapper and categorize by tensor type
  tensorWrapperMap_[tensorName] = tensorWrapper;

  if (QNN_TENSOR_GET_TYPE(nativeTensor) == QNN_TENSOR_TYPE_APP_WRITE) {
    inputTensorWrappers_.push_back(tensorWrapper);
  } else if (QNN_TENSOR_GET_TYPE(nativeTensor) == QNN_TENSOR_TYPE_APP_READ) {
    outputTensorWrappers_.push_back(tensorWrapper);
    // Record QNN output order (index in outputTensorWrappers_)
    qnnOutputNameToIndex_[tensorName] = static_cast<int>(outputTensorWrappers_.size() - 1);
  }

  return MODEL_NO_ERROR;
}

ModelError_t QNNModel::addTensor(const std::string& tensorName, Qnn_TensorType_t type, const Tensor& tensor,
                                 Qnn_QuantizeParams_t quantize) {
  auto tensorWrapper = QNNTensorWrapper::create(tensorName, type, tensor, quantize);
  return addTensorWrapper(tensorWrapper);
}

ModelError_t QNNModel::addStaticTensor(const std::string& tensorName, const Tensor& tensor, Qnn_QuantizeParams_t quantize) {
  auto tensorWrapper = QNNTensorWrapper::createStaticTensor(tensorName, tensor, quantize);
  return addTensorWrapper(tensorWrapper);
}

std::shared_ptr<QNNTensorWrapper> QNNModel::getTensorWrapper(const std::string& tensorName) {
  auto it = tensorWrapperMap_.find(tensorName);
  if (it != tensorWrapperMap_.end()) { return it->second; }
  return nullptr;
}

ModelError_t QNNModel::addNode(Qnn_OpConfigVersion_t version, const std::string& name, const std::string& packageName,
                               const std::string& type, const std::vector<std::shared_ptr<QNNParamTensorWrapper>>& tensorParams,
                               const std::vector<std::shared_ptr<QNNParamScalarWrapper>>& scalarParams,
                               const std::vector<std::string>& inputNames, const std::vector<std::string>& outputNames) {
  ModelError_t nodeError;
  Qnn_OpConfig_t opDefinition = QNN_OPCONFIG_INIT;
  opDefinition.version = version;
  VALIDATE_OP_CONFIG_VERSION((opDefinition), nodeError);

  // Store string parameters to ensure their lifetime
  nodeStringStorage_.push_back({name, packageName, type});
  const auto& storedStrings = nodeStringStorage_.back();

  // Store wrapper references for resource management
  for (auto& wrapper : tensorParams) { paramTensorWrappers_.push_back(wrapper); }
  for (auto& wrapper : scalarParams) { paramScalarWrappers_.push_back(wrapper); }

  // Prepare parameters
  size_t totalParams = tensorParams.size() + scalarParams.size();
  Qnn_Param_t* nodeParams = (Qnn_Param_t*)malloc(totalParams * sizeof(Qnn_Param_t));

  // Prepare input/output tensors
  Qnn_Tensor_t* inputs = (Qnn_Tensor_t*)malloc(inputNames.size() * sizeof(Qnn_Tensor_t));
  Qnn_Tensor_t* outputs = (Qnn_Tensor_t*)malloc(outputNames.size() * sizeof(Qnn_Tensor_t));

  if (nodeParams == nullptr || inputs == nullptr || outputs == nullptr) {
    MLLM_ERROR("QNNModel::addNode() failed to allocate memory for creating QNN OpConfig for node {}", storedStrings.name);
    freeMultiPtr(nodeParams, inputs, outputs);
    return MODEL_MEMORY_ALLOCATE_ERROR;
  }

  // Populate parameters
  uint32_t paramCounter = 0;
  for (auto& tensorParam : tensorParams) {
    auto nativeTensor = tensorParam->getNativeTensor();
    auto tensorName = QNN_TENSOR_GET_NAME(nativeTensor);

    if (qnnInterface_.tensorCreateGraphTensor(graph_, nativeTensor) != QNN_TENSOR_NO_ERROR) {
      MLLM_ERROR("QNNModel::addTensorWrapper() error creating tensor {}", tensorName);
      return MODEL_TENSOR_ERROR;
    }
    nodeParams[paramCounter++] = *tensorParam->getNativeParam();
  }

  for (auto& scalarParam : scalarParams) { nodeParams[paramCounter++] = *scalarParam->getNativeParam(); }

  // Populate input tensors
  size_t inputCounter = 0;
  for (const auto& inputName : inputNames) {
    auto inputWrapper = getTensorWrapper(inputName);
    if (!inputWrapper) {
      MLLM_ERROR("QNNModel::addNode() tensor {} not found on node {}", inputName, storedStrings.name);
      freeMultiPtr(nodeParams, inputs, outputs);
      return MODEL_TENSOR_ERROR;
    }
    inputs[inputCounter++] = *inputWrapper->getNativeTensor();
  }

  // Get output tensor wrappers and populate
  size_t outputCounter = 0;
  modelOutputTensorMap_[storedStrings.name] = {};
  for (const auto& outputName : outputNames) {
    auto outputWrapper = getTensorWrapper(outputName);
    if (!outputWrapper) {
      MLLM_ERROR("QNNModel::addNode() output tensor {} not found on node {}", outputName, storedStrings.name);
      freeMultiPtr(nodeParams, inputs, outputs);
      return MODEL_TENSOR_ERROR;
    }

    modelOutputTensorMap_[storedStrings.name].emplace_back(outputName);
    outputs[outputCounter++] = *outputWrapper->getNativeTensor();
  }

  // Define and add node to graph
  QNN_OP_CFG_SET_NAME(opDefinition, storedStrings.name.c_str());
  QNN_OP_CFG_SET_PACKAGE_NAME(opDefinition, storedStrings.packageName.c_str());
  QNN_OP_CFG_SET_TYPE_NAME(opDefinition, storedStrings.type.c_str());
  QNN_OP_CFG_SET_PARAMS(opDefinition, totalParams, nodeParams);
  QNN_OP_CFG_SET_INPUTS(opDefinition, inputNames.size(), inputs);
  QNN_OP_CFG_SET_OUTPUTS(opDefinition, outputNames.size(), outputs);

  if (doNodeValidations_) {
    auto validationStatus = qnnInterface_.backendValidateOpConfig(backendHandle_, opDefinition);
    if (validationStatus == QNN_BACKEND_ERROR_NOT_SUPPORTED) {
      MLLM_ERROR("QNNModel::addNode() validation API not supported.");
    } else if (validationStatus != QNN_SUCCESS) {
      MLLM_ERROR("QNNModel::addNode() validating node {} failed.", storedStrings.name);
      freeMultiPtr(nodeParams, inputs, outputs);
      return MODEL_GRAPH_ERROR;
    }
  }

  if (qnnInterface_.graphAddNode(graph_, opDefinition) != QNN_GRAPH_NO_ERROR) {
    MLLM_ERROR("QNNModel::addNode() adding node {} failed.", storedStrings.name);
    freeMultiPtr(nodeParams, inputs, outputs);
    return MODEL_GRAPH_ERROR;
  }

  freeMultiPtr(nodeParams, inputs, outputs);
  return MODEL_NO_ERROR;
}

ModelError_t QNNModel::finalizeGraph(Qnn_ProfileHandle_t profileHandle, Qnn_SignalHandle_t signalHandle) {
  if (graph_ == nullptr) {
    MLLM_ERROR("QNNModel::finalizeGraph() graph handle is null.");
    return MODEL_GRAPH_ERROR;
  }

  if (isFinalized_) {
    MLLM_WARN("QNNModel::finalizeGraph() graph {} is already finalized.", graphName_);
    return MODEL_NO_ERROR;
  }

  if (qnnInterface_.graphFinalize(graph_, profileHandle, signalHandle) != QNN_GRAPH_NO_ERROR) {
    MLLM_ERROR("QNNModel::finalizeGraph() finalizing graph {} failed.", graphName_);
    return MODEL_GRAPH_ERROR;
  }

  isFinalized_ = true;

  return MODEL_NO_ERROR;
}

ModelError_t QNNModel::freeCachedTensors() {
  ModelError_t err = MODEL_NO_ERROR;

  // Clear wrapper-based resources
  // Note: shared_ptr will automatically handle resource cleanup
  // Only clear non-input/output tensors to preserve graph interface
  for (auto it = tensorWrapperMap_.begin(); it != tensorWrapperMap_.end();) {
    auto wrapper = it->second;
    auto nativeTensor = wrapper->getNativeTensor();

    if (QNN_TENSOR_GET_TYPE(nativeTensor) != QNN_TENSOR_TYPE_APP_WRITE
        && QNN_TENSOR_GET_TYPE(nativeTensor) != QNN_TENSOR_TYPE_APP_READ) {
      it = tensorWrapperMap_.erase(it);
    } else {
      ++it;
    }
  }

  // Clear parameter wrappers (these are typically not needed after graph creation)
  paramTensorWrappers_.clear();
  paramScalarWrappers_.clear();
  return err;
}

size_t memscpy(void* dst, size_t dstSize, const void* src, size_t copySize) {
  if (!dst || !src || !dstSize || !copySize) return 0;

  size_t minSize = dstSize < copySize ? dstSize : copySize;

  memcpy(dst, src, minSize);

  return minSize;
}

ModelError_t getGraphInfoFromModel(QNNModel& model, GraphInfoPtr_t* graphInfoPtr) {
  ModelError_t err = MODEL_NO_ERROR;

  *graphInfoPtr = (GraphInfo_t*)malloc(sizeof(GraphInfo_t));
  auto graphInfo = *graphInfoPtr;
  if (graphInfo == nullptr) {
    MLLM_ERROR("getGraphInfoFromModels() graphsInfo malloc returned nullptr.");
    return MODEL_GRAPH_ERROR;
  }

  graphInfo->graph = model.getQnnGraph();
  graphInfo->graphName = strnDup(model.getQnnGraphName().c_str(), model.getQnnGraphName().size());
  if (graphInfo->graphName == nullptr) {
    MLLM_ERROR("getGraphInfoFromModels() failed to construct graphName. Received nullptr.");
    return MODEL_GRAPH_ERROR;
  }

  // allocate and add graph input/output tensors from wrappers
  auto inputWrappers = model.getGraphInputTensorWrappers();
  size_t numInputTensors = inputWrappers.size();
  size_t inputTensorsSize = numInputTensors * sizeof(Qnn_Tensor_t);
  graphInfo->inputTensors = (Qnn_Tensor_t*)malloc(inputTensorsSize);
  for (size_t i = 0; i < numInputTensors; ++i) {
    if (!deepCopyQnnTensorInfo(&graphInfo->inputTensors[i], inputWrappers[i]->getNativeTensor())) {
      MLLM_ERROR("getGraphInfoFromModel() failed to copy input tensor {}.", i);
      return MODEL_TENSOR_ERROR;
    }
  }
  graphInfo->numInputTensors = (uint32_t)numInputTensors;

  // allocate and add graph outputTensors
  auto outputWrappers = model.getGraphOutputTensorWrappers();
  size_t numOutputTensors = outputWrappers.size();
  size_t outputTensorsSize = numOutputTensors * sizeof(Qnn_Tensor_t);
  graphInfo->outputTensors = (Qnn_Tensor_t*)malloc(outputTensorsSize);
  for (size_t i = 0; i < numOutputTensors; ++i) {
    if (!deepCopyQnnTensorInfo(&graphInfo->outputTensors[i], outputWrappers[i]->getNativeTensor())) {
      MLLM_ERROR("getGraphInfoFromModel() failed to copy output tensor {}.", i);
      return MODEL_TENSOR_ERROR;
    }
  }
  graphInfo->numOutputTensors = (uint32_t)numOutputTensors;

  // graph composition is complete by this stage, free if any cached tensors remaining
  CALL_QNN(model.freeCachedTensors());
  return err;
}

}  // namespace mllm::qnn
