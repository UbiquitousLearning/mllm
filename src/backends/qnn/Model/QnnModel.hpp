//==============================================================================
//
//  Copyright (c) 2019-2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <limits>
#include <map>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "QnnLog.h"
#include "QnnModelPal.hpp"
#include "../WrapperUtils/QnnWrapperUtils.hpp"

namespace qnn_wrapper_api {

class QnnModel {
 public:
  ~QnnModel() = default;

  /**
   * @brief Creates a Qnn Graph within given context.
   *
   * @param[in] backendHandle A handle to the QNN backend handle which will be used to query the API
   *            symbols
   *
   * @param[in] qnnInterface the QNN backend interface to use
   *
   * @param[in] context A handler to the context where the model's graph would be created.
   *
   * @param[in] graphName The name to use for creating a graph in the context provided.
   *
   * @param[in] debug If flag is true, sets all tensors created in model to be
   *                     QNN_TENSOR_TYPE_APP_READ, essentially overwriting what is set
   *                  in Qnn_TensorType.
   *
   * @param[in] doNodeValidations If flag is set, all nodes added with addNode call
   *                              will be validated by Backend
   *
   * @param[in] graphConfigs Array of graph configurations to use for creating the QNN Graph.
   *            Default: nullptr
   *
   */
  ModelError_t initialize(const Qnn_BackendHandle_t& backendHandle,
                          const QNN_INTERFACE_VER_TYPE& qnnInterface,
                          const Qnn_ContextHandle_t& context,
                          const char* graphName,
                          bool debug,
                          uint8_t doNodeValidations              = 1,
                          const QnnGraph_Config_t** graphConfigs = nullptr);

  /**
   * @brief A wrapper function to create a tensor inside class's context graph.
   *
   * @param[in] nodeName Lookup name for node/layer
   *
   * @param[in] tensor A pointer to a struct containing information on the tensor
   *
   * @param[in] saveTensor Flag to indicate if tensor should be saved in object for later retrieval
   *                       with class getter functions.
   *
   * @return Error code
   *
   */
  ModelError_t addTensor(const char* nodeName, Qnn_Tensor_t* tensor, bool saveTensor = true);

  /**
   * @brief A wrapper function to create a tensor inside class's context graph.
   *
   * @param[in] nodeName Lookup name for node/layer
   *
   * @param[in] tensor A struct containing information on the tensor
   *
   * @param[in] saveTensor Flag to indicate if tensor should be saved in object for later retrieval
   *                       with class getter functions.
   *
   * @return Error code
   *
   */
  ModelError_t addTensor(const char* nodeName, Qnn_Tensor_t tensor, bool saveTensor = true);

  /**
   * @brief function to be used to query tensors created within this QnnModel instance
   *
   * @param[in] nodeName Lookup name for node/layer
   *
   * @param[in] tensorName Lookup name for tensor
   *
   * @param[out] tensor The corresponding Qnn_Tensor_t object for given tensor name.
   *
   * @return Error code
   *
   */
  ModelError_t getQnnTensor(const char*& nodeName, const char*& tensorName, Qnn_Tensor_t& tensor);
  ModelError_t getQnnTensor(std::string nodeName, std::string tensorName, Qnn_Tensor_t& tensor);

  /**
   * @brief A wrapper function to create a node in class's graph.
   *
   * @param[in] version The QNN version for Op_Config_t structure to use (e.g.
   * QNN_OPCONFIG_VERSION_1)
   *
   * @param[in] name The node name to use (e.g. my_graph_conv_1)
   *
   * @param[in] packageName The node package name (e.g. qti.aisw)
   *
   * @param[in] type The QNN_OP_QNN_OP_H node type (e.g. QNN_OP_ARGMAX)
   *
   * @param[in] params A struct object containing all the params for the node to be added. For
   * tensorParam case. The tensor will be created within the function and the data will be retrieved
   * from the binary blob to set the tensor data.
   *
   * @param[in] numOfParams The number of elements in above params object
   *
   * @param[in] inputNames List of tensor names for inputs to node. Note: the corresponding qnn
   * tensor objects must be created within this instance prior to being listed as input to a node
   *
   * @param[in] numOfInputs The number of elements in above inputNames object
   *
   * @param[in] outputTensors List of Qnn_Tensor_t objects for outputs from node.
   *                           Note1: the corresponding qnn tensor objects will be created in
   * function and must not already exist. Note2: the output names must be unique per graph
   *
   * @param[in] numOfOutputs The number of elements in above outputs object
   *
   * @return Error code
   *
   */
  ModelError_t addNode(Qnn_OpConfigVersion_t version,
                       const char* name,
                       const char* packageName,
                       const char* type,
                       Qnn_Param_t* params,
                       uint32_t numOfParams,
                       const char** inputNames,
                       uint32_t numOfInputs,
                       Qnn_Tensor_t* outputTensors,
                       uint32_t numOfOutputs);
  // overload for vector of inputNames
  ModelError_t addNode(Qnn_OpConfigVersion_t version,
                       const char* name,
                       const char* packageName,
                       const char* type,
                       Qnn_Param_t* params,
                       uint32_t numOfParams,
                       std::vector<std::string> inputNames,
                       uint32_t numOfInputs,
                       Qnn_Tensor_t* outputTensors,
                       uint32_t numOfOutputs);

  /**
   * @brief A wrapper function to return model's graph
   *
   * @return The Qnn graph object
   *
   */
  Qnn_GraphHandle_t getQnnGraph() { return m_graph; }

  /**
   * @brief A wrapper function to return model's graphName
   *
   * @return The Qnn graph object's name
   *
   */
  std::string getQnnGraphName() { return m_graphName; }

  /**
   * @brief A wrapper function to return model's graph input tensors
   *
   * @return vector of Qnn_Tensor_t objects
   *
   */
  std::vector<Qnn_Tensor_t> getGraphInputTensors() { return m_modelInputTensors; }

  /**
   * @brief A wrapper function to return model's graph output tensors
   *
   * @return vector of Qnn_Tensor_t objects
   *
   */
  std::vector<Qnn_Tensor_t> getGraphOutputTensors() { return m_modelOutputTensors; }

  /**
   * @brief A wrapper function to return graph's output tensors->op mapping
   *
   * @return map of std::string, std::vector<std::string>
   *
   */
  std::map<std::string, std::vector<std::string>> getOutputTensorMap() {
    return m_modelOutputTensorMap;
  }

  /**
   * @brief A wrapper function to finalize model's graph which includes calling backend finalize on
   * graph.
   *
   * @return Error code
   *
   */
  ModelError_t finalize(Qnn_ProfileHandle_t profile = nullptr, Qnn_SignalHandle_t signal = nullptr);

  /**
   * @brief Removes saved Qnn_Tensor_t objects and frees memory
   *        Note: Cleanup doesnt apply to input/output tensors as they are needed
   *        beyond this class finishes graph construction for the execute call. User of this API is
   *        expected to free those.
   *
   * @return Error code
   */
  ModelError_t freeCachedTensors();


  ModelError_t freeTensors();

  ModelError_t clearGraph();

 private:
  Qnn_GraphHandle_t m_graph = nullptr;
  std::string m_graphName;
  bool m_debug = false;  // flag to indicate if requested graph is to be run in debug mode
  // (i.e. all intermediate tensors will be accessible to client)
  // flag to indicate whether all addNode calls need to be validated
  bool m_doNodeValidations = true;

  std::vector<Qnn_Tensor_t> m_modelInputTensors;
  std::vector<Qnn_Tensor_t> m_modelOutputTensors;
  // keeps track of graph tensors to enable creating Qnn nodes from tensor names
  std::map<std::string, Qnn_Tensor_t> m_modelTensorsMap;
  std::map<std::string, std::vector<std::string>> m_modelOutputTensorMap;

  // Qnn Backend Interface Api
  QNN_INTERFACE_VER_TYPE m_qnnInterface;
  Qnn_BackendHandle_t m_backendHandle;

};  // QNN_MODEL_CLASS

/**
 * @brief A helper function to convert QnnModel objects to Graph struct for qnn_model c
 * interface
 * @param[in] models List of QnnModel objects
 * @param[in] numModels The number of elements in above models object
 *
 * @param[out] graphsInfo The corresponding array of Graph object for each of the above model
 * objects(note: this function will malloc memory needed to store the struct objects. Following free
 * shall be invoked when objects are no longer needed.
 *
 * @return Error code
 *
 */
ModelError_t getGraphInfoFromModels(QnnModel* models,
                                    uint32_t numModels,
                                    GraphInfoPtr_t** graphsInfo);
ModelError_t getSingleGraphInfoFromModel(QnnModel &model, GraphInfoPtr_t* graphInfoPtr);

/**
 * @brief A helper function to free memory malloced for communicating the Graph for a model(s)
 * @param[in] graphsInfo Pointer pointing to location of graph objects
 * @param[in] numGraphs The number of graph objects the above pointer is pointing to
 *
 * @return Error code
 *
 */
ModelError_t freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphs);
}  // namespace qnn_wrapper_api
