#pragma once

#include <map>
#include <string>
#include <vector>

#include "QNNUtils.hpp"
#include "QnnInterface.h"

namespace mllm {

typedef enum ModelError {
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
} ModelError_t;

class QNNModel {
public:
    ~QNNModel() = default;

    ModelError_t initialize(const Qnn_BackendHandle_t &backendHandle,
                            const QNN_INTERFACE_VER_TYPE &qnnInterface,
                            const Qnn_ContextHandle_t &context,
                            const char *graphName,
                            bool debug,
                            uint8_t doNodeValidations = 1,
                            const QnnGraph_Config_t **graphConfigs = nullptr);

    ModelError_t addTensor(const char *nodeName, Qnn_Tensor_t *tensor, bool saveTensor = true);

    ModelError_t addTensor(const char *nodeName, Qnn_Tensor_t tensor, bool saveTensor = true);

    ModelError_t getQnnTensor(std::string nodeName, std::string tensorName, Qnn_Tensor_t &tensor);

    ModelError_t addNode(Qnn_OpConfigVersion_t version,
                         const char *name,
                         const char *packageName,
                         const char *type,
                         std::vector<Qnn_Param_t> &params,
                         std::vector<std::string> inputNames,
                         std::vector<Qnn_Tensor_t> &outputTensors);

    Qnn_GraphHandle_t getQnnGraph() {
        return m_graph;
    }

    std::string getQnnGraphName() {
        return m_graphName;
    }

    std::vector<Qnn_Tensor_t> getGraphInputTensors() {
        return m_modelInputTensors;
    }

    std::vector<Qnn_Tensor_t> getGraphOutputTensors() {
        return m_modelOutputTensors;
    }

    std::map<std::string, std::vector<std::string>> getOutputTensorMap() {
        return m_modelOutputTensorMap;
    }

    ModelError_t freeCachedTensors();

private:
    Qnn_GraphHandle_t m_graph = nullptr;
    std::string m_graphName;
    bool m_debug = false; // flag to indicate if requested graph is to be run in debug mode
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

}; // QNN_MODEL_CLASS

// A helper function to convert QnnModel objects to Graph struct for qnn_model c
ModelError_t getSingleGraphInfoFromModel(QNNModel &model, GraphInfoPtr_t *graphInfoPtr);

} // namespace mllm
