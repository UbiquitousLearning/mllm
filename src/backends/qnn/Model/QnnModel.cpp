//==============================================================================
//
//  Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>

#include "QnnModel.hpp"
#include "QnnModelPal.hpp"
#include "QnnTypeMacros.hpp"
#include "Utils/QnnSampleAppUtils.hpp"

#define FREE_MEMORY(ptr1, ptr2, ptr3) \
  do {                                \
    free(ptr1);                       \
    free(ptr2);                       \
    free(ptr3);                       \
  } while (0)

namespace qnn_wrapper_api {

ModelError_t QnnModel::initialize(const Qnn_BackendHandle_t &backendHandle,
                                  const QNN_INTERFACE_VER_TYPE &qnnInterface,
                                  const Qnn_ContextHandle_t &context,
                                  const char *graphName,
                                  bool debug,
                                  uint8_t doNodeValidations,
                                  const QnnGraph_Config_t **graphConfigs) {
  if (backendHandle == nullptr) {
    PRINT_ERROR("QnnModel::initialize() nullptr passed as backend handle.");
    return MODEL_CONTEXT_ERROR;
  }
  if (context == nullptr) {
    PRINT_ERROR("QnnModel::initialize() nullptr passed as context handle.");
    return MODEL_CONTEXT_ERROR;
  }
  if (graphName == nullptr) {
    PRINT_ERROR("QnnModel::initialize() nullptr passed as graphName.");
    return MODEL_GRAPH_ERROR;
  }

  if (!m_graphName.empty()) {
    // only one graph is allowed per QnnModel
    PRINT_ERROR("QnnModel::initialize() model for graph %s already initialized.", graphName);
    return MODEL_GRAPH_ERROR;
  }

  if (!m_doNodeValidations) {
    PRINT_WARNING(
        "Node validation disabled. Backend will not perform op "
        "validation prior to adding Node. \n");
  }

  m_qnnInterface      = qnnInterface;
  m_backendHandle     = backendHandle;
  m_graphName         = graphName;
  m_debug             = debug;
  m_doNodeValidations = doNodeValidations;

  if (m_qnnInterface.graphCreate(context, graphName, graphConfigs, &m_graph) !=
          QNN_GRAPH_NO_ERROR ||
      m_graph == nullptr) {
    PRINT_ERROR("QnnModel::initialize() not able to create graph in given context.");
    return MODEL_GRAPH_ERROR;
  }

  return MODEL_NO_ERROR;
}

void QnnModel::setInitFromCache() {
  isFromCache = true;
}

ModelError_t QnnModel::addTensor(const char *nodeName, Qnn_Tensor_t *tensor, bool saveTensor) {
  ModelError_t err;
  if (!tensor) {
    PRINT_ERROR("QnnModel::addTensor() NULL tensor pointer provided.\n");
    return MODEL_TENSOR_ERROR;
  }
  VALIDATE_TENSOR_VERSION((*tensor), err);

  // Verify tensor being added is not a duplicate
  std::string mapEntry = std::string(QNN_TENSOR_GET_NAME(tensor));
  if (m_modelTensorsMap.find(mapEntry) != m_modelTensorsMap.end()) {
    PRINT_ERROR("QnnModel::addTensor() creating tensor %s for node %s. Tensor already exists.\n",
                mapEntry.c_str(),
                nodeName);

    return MODEL_TENSOR_ERROR;
  }

  const std::map<Qnn_DataType_t, size_t> dataTypeToSize = {
      {QNN_DATATYPE_INT_8, 1},
      {QNN_DATATYPE_INT_16, 2},
      {QNN_DATATYPE_INT_32, 4},
      {QNN_DATATYPE_INT_64, 8},
      {QNN_DATATYPE_UINT_8, 1},
      {QNN_DATATYPE_UINT_16, 2},
      {QNN_DATATYPE_UINT_32, 4},
      {QNN_DATATYPE_UINT_64, 8},
      {QNN_DATATYPE_FLOAT_16, 2},
      {QNN_DATATYPE_FLOAT_32, 4},
      {QNN_DATATYPE_BOOL_8, 1},
      {QNN_DATATYPE_SFIXED_POINT_8, 1},
      {QNN_DATATYPE_SFIXED_POINT_16, 2},
      {QNN_DATATYPE_SFIXED_POINT_32, 4},
      {QNN_DATATYPE_UFIXED_POINT_8, 1},
      {QNN_DATATYPE_UFIXED_POINT_16, 2},
      {QNN_DATATYPE_UFIXED_POINT_32, 4},
  };

  if (dataTypeToSize.find(QNN_TENSOR_GET_DATA_TYPE(tensor)) == dataTypeToSize.end()) {
    PRINT_ERROR(
        "QnnModel::addTensor() invalid QNN data type provided, %u, for tensor %s on node %s\n",
        QNN_TENSOR_GET_DATA_TYPE(tensor),
        QNN_TENSOR_GET_NAME(tensor),
        nodeName);
    return MODEL_TENSOR_ERROR;
  }

  // sanity check tensor data if addTensor used for static tensor
  if (QNN_TENSOR_GET_TYPE(tensor) == QNN_TENSOR_TYPE_STATIC) {
    if (QNN_TENSOR_GET_MEM_TYPE(tensor) != QNN_TENSORMEMTYPE_RAW) {
      PRINT_ERROR(
          "QnnModel::addTensor(): Expected raw memType in provided static tensor %s for node %s",
          mapEntry.c_str(),
          nodeName);
      return MODEL_TENSOR_ERROR;
    }
    // verify size expressed by the dims matches the raw tensor size
    uint32_t qnnTensorSize =
        std::accumulate(QNN_TENSOR_GET_DIMENSIONS(tensor),
                        QNN_TENSOR_GET_DIMENSIONS(tensor) + QNN_TENSOR_GET_RANK(tensor),
                        (uint32_t)dataTypeToSize.find(QNN_TENSOR_GET_DATA_TYPE(tensor))->second,
                        std::multiplies<uint32_t>());
    if (qnnTensorSize != QNN_TENSOR_GET_CLIENT_BUF(tensor).dataSize) {
      PRINT_ERROR(
          "QnnModel::addTensor(): Adding STATIC tensor, length mismatch between clientBuf"
          "size and tensor Dims(dim * rank * sizeof(datatype) for, nodeName: %s, tensorName: %s."
          "Got tensorSize: %d, tensor.clientBuf.dataSize: %d.\n",
          nodeName,
          QNN_TENSOR_GET_NAME(tensor),
          qnnTensorSize,
          QNN_TENSOR_GET_CLIENT_BUF(tensor).dataSize);
      return MODEL_TENSOR_ERROR;
    }
  }

  if (m_debug && QNN_TENSOR_GET_TYPE(tensor) == QNN_TENSOR_TYPE_NATIVE) {
    // for debug, make all tensors accessible by client
    QNN_TENSOR_SET_TYPE(tensor, QNN_TENSOR_TYPE_APP_READ);
  }

  if (!isFromCache) {
      if (m_qnnInterface.tensorCreateGraphTensor(m_graph, tensor) != QNN_TENSOR_NO_ERROR) {
          PRINT_ERROR("QnnModel::addTensor() Creating tensor for node: %s, tensorName: %s.\n",
                      nodeName,
                      QNN_TENSOR_GET_NAME(tensor));
          return MODEL_TENSOR_ERROR;
      }
  }

  if (saveTensor) {
    Qnn_Tensor_t tensorCopy;
    if (!qnn::tools::sample_app::deepCopyQnnTensorInfo(&tensorCopy, tensor)) {
      return MODEL_TENSOR_ERROR;
    }

    // save network input/outputs tensors to use for setting the Qnn graph's input and output
    // tensors for populating GraphInfo_t for caller
    if (QNN_TENSOR_GET_TYPE(tensor) == QNN_TENSOR_TYPE_APP_WRITE) {
      m_modelInputTensors.push_back(tensorCopy);
    } else if (QNN_TENSOR_GET_TYPE(tensor) == QNN_TENSOR_TYPE_APP_READ) {
      m_modelOutputTensors.push_back(tensorCopy);
    }

    // save created tensors for later lookup to populate graph node construction
    m_modelTensorsMap[mapEntry] = tensorCopy;
  }

  return MODEL_NO_ERROR;
}

ModelError_t QnnModel::addTensor(const char *nodeName, Qnn_Tensor_t tensor, bool saveTensor) {
  return addTensor(nodeName, &tensor, saveTensor);
}

ModelError_t QnnModel::getQnnTensor(const char *&nodeName,
                                    const char *&tensorName,
                                    Qnn_Tensor_t &tensor) {
  std::string mapEntry = std::string(tensorName);
  if (m_modelTensorsMap.find(tensorName) == m_modelTensorsMap.end()) {
    PRINT_ERROR(
        "QnnModel::getQnnTensor() tensor %s not found on node %s\n", mapEntry.c_str(), nodeName);
    return MODEL_TENSOR_ERROR;
  }
  tensor = m_modelTensorsMap[mapEntry];

  return MODEL_NO_ERROR;
}

// overload for string tensorName
ModelError_t QnnModel::getQnnTensor(std::string nodeName,
                                    std::string tensorName,
                                    Qnn_Tensor_t &tensor) {
  if (m_modelTensorsMap.find(tensorName) == m_modelTensorsMap.end()) {
    PRINT_ERROR(
        "QnnModel::getQnnTensor() tensor %s not found on node %s\n", tensorName.c_str(), nodeName.c_str());
    return MODEL_TENSOR_ERROR;
  }
  tensor = m_modelTensorsMap[tensorName];

  return MODEL_NO_ERROR;
}


ModelError_t QnnModel::addNode(Qnn_OpConfigVersion_t version,
                               const char *name,
                               const char *packageName,
                               const char *type,
                               Qnn_Param_t *params,
                               uint32_t numOfParams,
                               const char **inputNames,
                               uint32_t numOfInputs,
                               Qnn_Tensor_t *outputTensors,
                               uint32_t numOfOutputs) {
  ModelError_t nodeError;
  Qnn_OpConfig_t opDefinition = QNN_OPCONFIG_INIT;
  opDefinition.version        = version;
  VALIDATE_OP_CONFIG_VERSION((opDefinition), nodeError);

  // populate Qnn param for node
  Qnn_Param_t *nodeParams = (Qnn_Param_t *)malloc(numOfParams * sizeof(Qnn_Param_t));

  // populate input tensors for node
  Qnn_Tensor_t *inputs = (Qnn_Tensor_t *)malloc(numOfInputs * sizeof(Qnn_Tensor_t));

  // populate output tensors of node
  Qnn_Tensor_t *outputs = (Qnn_Tensor_t *)malloc(numOfOutputs * sizeof(Qnn_Tensor_t));

  if (nodeParams == nullptr || inputs == nullptr || outputs == nullptr) {
    PRINT_ERROR(
        "QnnModel::addNode() failed for allocate memory for creating QNN OpConfig for node %s.\n",
        name);
    FREE_MEMORY(nodeParams, inputs, outputs);
    return MODEL_MEMORY_ALLOCATE_ERROR;
  }
  uint32_t nodeParamsCounter = 0;
  for (size_t i = 0; i < numOfParams; i++) {
    switch (params[i].paramType) {
      case QNN_PARAMTYPE_TENSOR: {
        Qnn_Tensor_t &tensor = params[i].tensorParam;
        // Note: set saveTensor to false as no need to save tensor beyond this
        //         function call for params
        nodeError = addTensor(name, &tensor, false);
        if (nodeError != MODEL_NO_ERROR) {
          PRINT_ERROR("QnnModel::addNode() addTensor() failed for tensor param %s on node %s.\n",
                      QNN_TENSOR_GET_NAME(tensor),
                      name);
          FREE_MEMORY(nodeParams, inputs, outputs);
          return nodeError;
        }
        nodeParams[nodeParamsCounter].paramType     = QNN_PARAMTYPE_TENSOR;
        nodeParams[nodeParamsCounter].name          = params[i].name;
        nodeParams[nodeParamsCounter++].tensorParam = tensor;
        break;
      }
      case QNN_PARAMTYPE_SCALAR: {
        nodeParams[nodeParamsCounter].paramType     = QNN_PARAMTYPE_SCALAR;
        nodeParams[nodeParamsCounter].name          = params[i].name;
        nodeParams[nodeParamsCounter++].scalarParam = params[i].scalarParam;
        break;
      }
      default: {
        PRINT_ERROR("QnnModel::addNode() unknown param type passed for param %s on node %s.\n",
                    params[i].name,
                    name);
        FREE_MEMORY(nodeParams, inputs, outputs);
        return MODEL_PARAMS_ERROR;
      }
    }
  }

  size_t inputsCounter = 0;
  for (size_t j = 0; j < numOfInputs; j++) {
    nodeError = getQnnTensor(name, inputNames[j], inputs[inputsCounter++]);
    if (nodeError != MODEL_NO_ERROR) {
      PRINT_ERROR("QnnModel::addNode() getQnnTensor() failed for tensor %s on node %s.\n",
                  inputNames[j],
                  name);
      FREE_MEMORY(nodeParams, inputs, outputs);
      return nodeError;
    }
  }

  size_t outputsCounter        = 0;
  m_modelOutputTensorMap[name] = {};
  for (size_t k = 0; k < numOfOutputs; k++) {
    // create node output tensors first
    nodeError = addTensor(name, outputTensors[k]);
    if (nodeError != MODEL_NO_ERROR) {
      PRINT_ERROR("QnnModel::addNode() addTensor() failed for tensor %s on node %s\n",
                  QNN_TENSOR_GET_NAME(outputTensors[k]),
                  name);
      FREE_MEMORY(nodeParams, inputs, outputs);
      return nodeError;
    }
    const char *outTensorName = QNN_TENSOR_GET_NAME(outputTensors[k]);
    m_modelOutputTensorMap[name].push_back(outTensorName);
    nodeError = getQnnTensor(name, outTensorName, outputs[outputsCounter++]);
    if (nodeError != MODEL_NO_ERROR) {
      PRINT_ERROR("QnnModel::addNode() getQnnTensor() failed for tensor %s on node %s.\n",
                  outTensorName,
                  name);
      FREE_MEMORY(nodeParams, inputs, outputs);
      return nodeError;
    }
  }

  // define and add node to graph
  QNN_OP_CFG_SET_NAME(opDefinition, name);
  QNN_OP_CFG_SET_PACKAGE_NAME(opDefinition, packageName);
  QNN_OP_CFG_SET_TYPE_NAME(opDefinition, type);
  QNN_OP_CFG_SET_PARAMS(opDefinition, numOfParams, nodeParams);
  QNN_OP_CFG_SET_INPUTS(opDefinition, numOfInputs, inputs);
  QNN_OP_CFG_SET_OUTPUTS(opDefinition, numOfOutputs, outputs);

  if (m_doNodeValidations) {
    auto validationStatus = m_qnnInterface.backendValidateOpConfig(m_backendHandle, opDefinition);
    if (validationStatus == QNN_BACKEND_ERROR_NOT_SUPPORTED) {
      PRINT_DEBUG("QnnModel::addNode() validation API not supported.\n");
    } else if (validationStatus != QNN_SUCCESS) {
      PRINT_ERROR("QnnModel::addNode() validating node %s failed.\n", name);
      FREE_MEMORY(nodeParams, inputs, outputs);
      return MODEL_GRAPH_ERROR;
    }
  }

  if (m_qnnInterface.graphAddNode(m_graph, opDefinition) != QNN_GRAPH_NO_ERROR) {
    PRINT_ERROR("QnnModel::addNode() adding node %s failed.\n", name);
    FREE_MEMORY(nodeParams, inputs, outputs);
    return MODEL_GRAPH_ERROR;
  }

  FREE_MEMORY(nodeParams, inputs, outputs);
  return MODEL_NO_ERROR;
}

// overload for string tensorName
ModelError_t QnnModel::addNode(Qnn_OpConfigVersion_t version,
                               const char *name,
                               const char *packageName,
                               const char *type,
                               Qnn_Param_t *params,
                               uint32_t numOfParams,
                               std::vector<std::string> inputNames,
                               uint32_t numOfInputs,
                               Qnn_Tensor_t *outputTensors,
                               uint32_t numOfOutputs) {
  ModelError_t nodeError;
  Qnn_OpConfig_t opDefinition = QNN_OPCONFIG_INIT;
  opDefinition.version        = version;
  VALIDATE_OP_CONFIG_VERSION((opDefinition), nodeError);

  // populate Qnn param for node
  Qnn_Param_t *nodeParams = (Qnn_Param_t *)malloc(numOfParams * sizeof(Qnn_Param_t));

  // populate input tensors for node
  Qnn_Tensor_t *inputs = (Qnn_Tensor_t *)malloc(numOfInputs * sizeof(Qnn_Tensor_t));

  // populate output tensors of node
  Qnn_Tensor_t *outputs = (Qnn_Tensor_t *)malloc(numOfOutputs * sizeof(Qnn_Tensor_t));

  if (nodeParams == nullptr || inputs == nullptr || outputs == nullptr) {
    PRINT_ERROR(
        "QnnModel::addNode() failed for allocate memory for creating QNN OpConfig for node %s.\n",
        name);
    FREE_MEMORY(nodeParams, inputs, outputs);
    return MODEL_MEMORY_ALLOCATE_ERROR;
  }
  uint32_t nodeParamsCounter = 0;
  for (size_t i = 0; i < numOfParams; i++) {
    switch (params[i].paramType) {
      case QNN_PARAMTYPE_TENSOR: {
        Qnn_Tensor_t &tensor = params[i].tensorParam;
        // Note: set saveTensor to false as no need to save tensor beyond this
        //         function call for params
        nodeError = addTensor(name, &tensor, false);
        if (nodeError != MODEL_NO_ERROR) {
          PRINT_ERROR("QnnModel::addNode() addTensor() failed for tensor param %s on node %s.\n",
                      QNN_TENSOR_GET_NAME(tensor),
                      name);
          FREE_MEMORY(nodeParams, inputs, outputs);
          return nodeError;
        }
        nodeParams[nodeParamsCounter].paramType     = QNN_PARAMTYPE_TENSOR;
        nodeParams[nodeParamsCounter].name          = params[i].name;
        nodeParams[nodeParamsCounter++].tensorParam = tensor;
        break;
      }
      case QNN_PARAMTYPE_SCALAR: {
        nodeParams[nodeParamsCounter].paramType     = QNN_PARAMTYPE_SCALAR;
        nodeParams[nodeParamsCounter].name          = params[i].name;
        nodeParams[nodeParamsCounter++].scalarParam = params[i].scalarParam;
        break;
      }
      default: {
        PRINT_ERROR("QnnModel::addNode() unknown param type passed for param %s on node %s.\n",
                    params[i].name,
                    name);
        FREE_MEMORY(nodeParams, inputs, outputs);
        return MODEL_PARAMS_ERROR;
      }
    }
  }

  size_t inputsCounter = 0;
  for (size_t j = 0; j < numOfInputs; j++) {
    nodeError = getQnnTensor(name, inputNames[j], inputs[inputsCounter++]);
    if (nodeError != MODEL_NO_ERROR) {
      PRINT_ERROR("QnnModel::addNode() getQnnTensor() failed for tensor %s on node %s.\n",
                  inputNames[j].c_str(),
                  name);
      FREE_MEMORY(nodeParams, inputs, outputs);
      return nodeError;
    }
  }

  size_t outputsCounter        = 0;
  m_modelOutputTensorMap[name] = {};
  for (size_t k = 0; k < numOfOutputs; k++) {
    // create node output tensors first
    nodeError = addTensor(name, outputTensors[k]);
    if (nodeError != MODEL_NO_ERROR) {
      PRINT_ERROR("QnnModel::addNode() addTensor() failed for tensor %s on node %s\n",
                  QNN_TENSOR_GET_NAME(outputTensors[k]),
                  name);
      FREE_MEMORY(nodeParams, inputs, outputs);
      return nodeError;
    }
    const char *outTensorName = QNN_TENSOR_GET_NAME(outputTensors[k]);
    m_modelOutputTensorMap[name].push_back(outTensorName);
    nodeError = getQnnTensor(name, outTensorName, outputs[outputsCounter++]);
    if (nodeError != MODEL_NO_ERROR) {
      PRINT_ERROR("QnnModel::addNode() getQnnTensor() failed for tensor %s on node %s.\n",
                  outTensorName,
                  name);
      FREE_MEMORY(nodeParams, inputs, outputs);
      return nodeError;
    }
  }

  // define and add node to graph
  QNN_OP_CFG_SET_NAME(opDefinition, name);
  QNN_OP_CFG_SET_PACKAGE_NAME(opDefinition, packageName);
  QNN_OP_CFG_SET_TYPE_NAME(opDefinition, type);
  QNN_OP_CFG_SET_PARAMS(opDefinition, numOfParams, nodeParams);
  QNN_OP_CFG_SET_INPUTS(opDefinition, numOfInputs, inputs);
  QNN_OP_CFG_SET_OUTPUTS(opDefinition, numOfOutputs, outputs);

  if (m_doNodeValidations) {
    auto validationStatus = m_qnnInterface.backendValidateOpConfig(m_backendHandle, opDefinition);
    if (validationStatus == QNN_BACKEND_ERROR_NOT_SUPPORTED) {
      PRINT_DEBUG("QnnModel::addNode() validation API not supported.\n");
    } else if (validationStatus != QNN_SUCCESS) {
      PRINT_ERROR("QnnModel::addNode() validating node %s failed.\n", name);
      FREE_MEMORY(nodeParams, inputs, outputs);
      return MODEL_GRAPH_ERROR;
    }
  }

  if (m_qnnInterface.graphAddNode(m_graph, opDefinition) != QNN_GRAPH_NO_ERROR) {
    PRINT_ERROR("QnnModel::addNode() adding node %s failed.\n", name);
    FREE_MEMORY(nodeParams, inputs, outputs);
    return MODEL_GRAPH_ERROR;
  }

  FREE_MEMORY(nodeParams, inputs, outputs);
  return MODEL_NO_ERROR;
}

ModelError_t QnnModel::freeCachedTensors() {
  ModelError_t err = MODEL_NO_ERROR;

  // cleanup cached tensors
  for (std::map<std::string, Qnn_Tensor_t>::iterator tensorIt = m_modelTensorsMap.begin();
       tensorIt != m_modelTensorsMap.end();) {
    Qnn_Tensor_t &tensor = tensorIt->second;
    if (QNN_TENSOR_GET_TYPE(tensor) != QNN_TENSOR_TYPE_APP_WRITE &&
        QNN_TENSOR_GET_TYPE(tensor) != QNN_TENSOR_TYPE_APP_READ) {
      VALIDATE(freeQnnTensor(tensor), err);
      tensorIt = m_modelTensorsMap.erase(tensorIt);
    } else {
      tensorIt++;
    }
  }

  return err;
}

ModelError_t QnnModel::finalize(Qnn_ProfileHandle_t profile, Qnn_SignalHandle_t signal) {
  ModelError_t err;

  // finalize the graph
  if (m_qnnInterface.graphFinalize(m_graph, profile, signal) != QNN_GRAPH_NO_ERROR) {
    PRINT_ERROR("QnnModel::finalize() finalizing graph failed.\n");
    return MODEL_GRAPH_ERROR;
  }

  VALIDATE(freeCachedTensors(), err);

  return err;
}

size_t memscpy(void *dst, size_t dstSize, const void *src, size_t copySize) {
    if (!dst || !src || !dstSize || !copySize) return 0;

    size_t minSize = dstSize < copySize ? dstSize : copySize;

    memcpy(dst, src, minSize);

    return minSize;
}

ModelError_t getGraphInfoFromModels(QnnModel *models,
                                    uint32_t numModels,
                                    GraphInfoPtr_t **graphsInfo) {
  ModelError_t err = MODEL_NO_ERROR;
  if (models == nullptr || graphsInfo == nullptr || numModels <= 0) {
    PRINT_ERROR(
        "getGraphInfoFromModels() models and graphsInfo uninitialized or number of models is "
        "<= 0.\n");
    return MODEL_GRAPH_ERROR;
  }

  *graphsInfo = (GraphInfo_t **)malloc(numModels * sizeof(GraphInfo_t *));
  if (*graphsInfo == nullptr) {
    PRINT_ERROR("getGraphInfoFromModels() graphsInfo malloc returned nullptr.\n");
    return MODEL_GRAPH_ERROR;
  }

  GraphInfo_t *graphArr = (GraphInfo_t *)malloc(numModels * sizeof(GraphInfo_t));
  if (graphArr == nullptr) {
    PRINT_ERROR("getGraphInfoFromModels() graphArr malloc returned nullptr.\n");
    return MODEL_GRAPH_ERROR;
  }

  for (uint32_t i = 0; i < numModels; i++) {
    QnnModel &model   = models[i];
    graphArr[i].graph = model.getQnnGraph();
    graphArr[i].graphName =
        strnDup(model.getQnnGraphName().c_str(), model.getQnnGraphName().size());
    if (graphArr[i].graphName == nullptr) {
      PRINT_ERROR("getGraphInfoFromModels() failed to construct graphName. Received nullptr.\n");
      return MODEL_GRAPH_ERROR;
    }

    // allocate and add graph input/output TensorsWrapper. Note: no need to make deep copies of
    // the tensor's pointer members as they are already allocated on heap in the addTensor
    // function call.
    std::vector<Qnn_Tensor_t> graphInputTensors = model.getGraphInputTensors();
    size_t numInputTensors                      = graphInputTensors.size();
    size_t inputTensorsSize                     = numInputTensors * sizeof(Qnn_Tensor_t);
    graphArr[i].inputTensors                    = (Qnn_Tensor_t *)malloc(inputTensorsSize);
    memscpy(graphArr[i].inputTensors, inputTensorsSize, graphInputTensors.data(), inputTensorsSize);
    graphArr[i].numInputTensors = (uint32_t)numInputTensors;
    // allocate and add graph outputTensors
    std::vector<Qnn_Tensor_t> graphOutputTensors = model.getGraphOutputTensors();
    size_t numOutputTensors                      = graphOutputTensors.size();
    size_t outputTensorsSize                     = numOutputTensors * sizeof(Qnn_Tensor_t);
    graphArr[i].outputTensors                    = (Qnn_Tensor_t *)malloc(outputTensorsSize);
    memscpy(
        graphArr[i].outputTensors, outputTensorsSize, graphOutputTensors.data(), outputTensorsSize);
    graphArr[i].numOutputTensors = (uint32_t)numOutputTensors;

    // have return object point to the populated graph struct
    (*graphsInfo)[i] = graphArr + i;

    // graph composition is complete by this stage, free if any cached tensors remaining
    VALIDATE(model.freeCachedTensors(), err);
  }

  return err;
}

ModelError_t getSingleGraphInfoFromModel(QnnModel &model, GraphInfoPtr_t* graphInfoPtr) {
    ModelError_t err = MODEL_NO_ERROR;

    *graphInfoPtr = (GraphInfo_t *)malloc(sizeof(GraphInfo_t));
    auto graphInfo = *graphInfoPtr;
    if (graphInfo == nullptr) {
        PRINT_ERROR("getGraphInfoFromModels() graphsInfo malloc returned nullptr.\n");
        return MODEL_GRAPH_ERROR;
    }

    graphInfo->graph = model.getQnnGraph();
    graphInfo->graphName =
        strnDup(model.getQnnGraphName().c_str(), model.getQnnGraphName().size());
    if (graphInfo->graphName == nullptr) {
        PRINT_ERROR("getGraphInfoFromModels() failed to construct graphName. Received nullptr.\n");
        return MODEL_GRAPH_ERROR;
    }

    // allocate and add graph input/output TensorsWrapper. Note: no need to make deep copies of
    // the tensor's pointer members as they are already allocated on heap in the addTensor
    // function call.
    std::vector<Qnn_Tensor_t> graphInputTensors = model.getGraphInputTensors();
    size_t numInputTensors = graphInputTensors.size();
    size_t inputTensorsSize = numInputTensors * sizeof(Qnn_Tensor_t);
    graphInfo->inputTensors = (Qnn_Tensor_t *)malloc(inputTensorsSize);
    memscpy(graphInfo->inputTensors, inputTensorsSize, graphInputTensors.data(), inputTensorsSize);
    graphInfo->numInputTensors = (uint32_t)numInputTensors;
    // allocate and add graph outputTensors
    std::vector<Qnn_Tensor_t> graphOutputTensors = model.getGraphOutputTensors();
    size_t numOutputTensors = graphOutputTensors.size();
    size_t outputTensorsSize = numOutputTensors * sizeof(Qnn_Tensor_t);
    graphInfo->outputTensors = (Qnn_Tensor_t *)malloc(outputTensorsSize);
    memscpy(
        graphInfo->outputTensors, outputTensorsSize, graphOutputTensors.data(), outputTensorsSize);
    graphInfo->numOutputTensors = (uint32_t)numOutputTensors;

    // graph composition is complete by this stage, free if any cached tensors remaining
    VALIDATE(model.freeCachedTensors(), err);
    return err;
}

ModelError_t QnnModel::freeTensors() {
  
  for (std::map<std::string, Qnn_Tensor_t>::iterator tensorIt = m_modelTensorsMap.begin();
       tensorIt != m_modelTensorsMap.end();) {
    Qnn_Tensor_t &tensor = tensorIt->second;
    
    tensorIt = m_modelTensorsMap.erase(tensorIt++);
  }

  return MODEL_NO_ERROR;
}

ModelError_t QnnModel::clearGraph() {
  
  m_modelInputTensors.resize(0);
  m_modelOutputTensors.resize(0);

  m_modelOutputTensorMap.clear();
  m_graphName.clear();
  
  

  return MODEL_NO_ERROR;
}

}  // namespace qnn_wrapper_api