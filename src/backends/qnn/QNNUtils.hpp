
#pragma once

#include "Log.h"
#include "QnnCommon.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

namespace mllm {

#define CALL_QNN(apiCall)                                                \
    do {                                                                 \
        int errorCode = ((apiCall) & 0xFFFF);                            \
        if (errorCode != QNN_SUCCESS) {                                  \
            MLLM_LOG_ERROR("Error in file %s, line %d: error code %d\n", \
                           __FILE__, __LINE__, errorCode);               \
            assert(errorCode == QNN_SUCCESS);                            \
        }                                                                \
    } while (0)

// func def for loading QNN Interface
typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t ***providerList,
                                                          uint32_t *numProviders);
// func def of loading QNN System Interface
typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(
    const QnnSystemInterface_t ***providerList, uint32_t *numProviders);

extern QnnInterfaceGetProvidersFn_t QnnInterface_getProviders;
extern QnnSystemInterfaceGetProvidersFn_t QnnSystemInterface_getProviders;

bool loadQNNSymbol();
bool loadQNNSystemSymbol();

// Utils for copying metadata to GraphInfo
typedef struct GraphInfo {
    Qnn_GraphHandle_t graph;
    char *graphName;
    Qnn_Tensor_t *inputTensors;
    uint32_t numInputTensors;
    Qnn_Tensor_t *outputTensors;
    uint32_t numOutputTensors;
} GraphInfo_t;
typedef GraphInfo_t *GraphInfoPtr_t;

typedef struct GraphConfigInfo {
    char *graphName;
    const QnnGraph_Config_t **graphConfigs;
} GraphConfigInfo_t;

bool copyMetadataToGraphsInfo(const QnnSystemContext_BinaryInfo_t *binaryInfo,
                              GraphInfo_t **&graphsInfo,
                              uint32_t &graphsCount);

bool copyGraphsInfo(const QnnSystemContext_GraphInfo_t *graphsInput,
                    const uint32_t numGraphs,
                    GraphInfo_t **&graphsInfo);

bool copyGraphsInfoV1(const QnnSystemContext_GraphInfoV1_t *graphInfoSrc,
                      GraphInfo_t *graphInfoDst);

bool copyGraphsInfoV3(const QnnSystemContext_GraphInfoV3_t *graphInfoSrc,
                      GraphInfo_t *graphInfoDst);

bool copyTensorsInfo(const Qnn_Tensor_t *tensorsInfoSrc,
                     Qnn_Tensor_t *&tensorWrappers,
                     uint32_t tensorsCount);

bool deepCopyQnnTensorInfo(Qnn_Tensor_t *dst, const Qnn_Tensor_t *src);

bool freeQnnTensor(Qnn_Tensor_t &tensor);

bool freeQnnTensors(Qnn_Tensor_t *&tensors, uint32_t numTensors);

} // namespace mllm