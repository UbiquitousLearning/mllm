#include "QNNUtils.hpp"
#include "Log.h"
#include "QnnTypeMacros.hpp"
#include <cstdint>
#include <dlfcn.h>

namespace mllm {

QnnInterfaceGetProvidersFn_t QnnInterface_getProviders = nullptr;

bool loadQNNSymbol() {
    MLLM_LOG_INFO_STREAM << "QNN Backend Lib: libQnnHtp.so";
    void *qnnLibHandle = nullptr;
    qnnLibHandle = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
    const char *errorOpen = dlerror();
    if (!qnnLibHandle) {
        MLLM_LOG_ERROR_LEGACY("Failed to open QNN libs. Ensure that the libs related to the QNN HTP backend is available in your environment. dlerror() returns %s.\n", errorOpen);
        return false;
    }

    QnnInterface_getProviders = (QnnInterfaceGetProvidersFn_t)dlsym(qnnLibHandle, "QnnInterface_getProviders");
    const char *errorSym = dlerror();
    if (!QnnInterface_getProviders) {
        MLLM_LOG_ERROR_LEGACY("Failed to load symbol <QnnInterface_getProviders>. dlerror returns %s.\n", errorSym);
        dlclose(qnnLibHandle);
        return false;
    }

    return true;
}

QnnSystemInterfaceGetProvidersFn_t QnnSystemInterface_getProviders = nullptr;

bool loadQNNSystemSymbol() {
    void *systemLibraryHandle = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
    const char *errorOpen = dlerror();
    if (!systemLibraryHandle) {
        MLLM_LOG_ERROR_LEGACY("Failed to open QNN System libs. Ensure that the libs related to the QNN System backend is available in your environment. dlerror() returns %s.\n", errorOpen);
        return false;
    }

    QnnSystemInterface_getProviders = (QnnSystemInterfaceGetProvidersFn_t)dlsym(systemLibraryHandle, "QnnSystemInterface_getProviders");
    const char *errorSym = dlerror();
    if (!QnnSystemInterface_getProviders) {
        MLLM_LOG_ERROR_LEGACY("Failed to load symbol <QnnSystemInterface_getProviders>. dlerror returns %s.\n", errorSym);
        dlclose(systemLibraryHandle);
        return false;
    }

    return true;
}

bool copyMetadataToGraphsInfo(const QnnSystemContext_BinaryInfo_t *binaryInfo,
                              GraphInfo_t **&graphsInfo,
                              uint32_t &graphsCount) {
    if (nullptr == binaryInfo) {
        MLLM_LOG_ERROR("binaryInfo is nullptr.");
        return false;
    }
    graphsCount = 0;
    if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        if (binaryInfo->contextBinaryInfoV1.graphs) {
            if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV1.graphs,
                                binaryInfo->contextBinaryInfoV1.numGraphs,
                                graphsInfo)) {
                MLLM_LOG_ERROR("Failed while copying graphs Info.");
                return false;
            }
            graphsCount = binaryInfo->contextBinaryInfoV1.numGraphs;
            return true;
        }
    } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        if (binaryInfo->contextBinaryInfoV2.graphs) {
            if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV2.graphs,
                                binaryInfo->contextBinaryInfoV2.numGraphs,
                                graphsInfo)) {
                MLLM_LOG_ERROR("Failed while copying graphs Info.");
                return false;
            }
            graphsCount = binaryInfo->contextBinaryInfoV2.numGraphs;
            return true;
        }
    } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        if (binaryInfo->contextBinaryInfoV3.graphs) {
            if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV3.graphs,
                                binaryInfo->contextBinaryInfoV3.numGraphs,
                                graphsInfo)) {
                MLLM_LOG_ERROR("Failed while copying graphs Info.");
                return false;
            }
            graphsCount = binaryInfo->contextBinaryInfoV3.numGraphs;
            return true;
        }
    }
    MLLM_LOG_ERROR("Unrecognized system context binary info version.");
    return false;
}

bool copyGraphsInfoV1(const QnnSystemContext_GraphInfoV1_t *graphInfoSrc,
                      GraphInfo_t *graphInfoDst) {
    graphInfoDst->graphName = nullptr;
    if (graphInfoSrc->graphName) {
        graphInfoDst->graphName =
            strndup(graphInfoSrc->graphName, strlen(graphInfoSrc->graphName));
    }
    graphInfoDst->inputTensors = nullptr;
    graphInfoDst->numInputTensors = 0;
    if (graphInfoSrc->graphInputs) {
        if (!copyTensorsInfo(
                graphInfoSrc->graphInputs, graphInfoDst->inputTensors, graphInfoSrc->numGraphInputs)) {
            return false;
        }
        graphInfoDst->numInputTensors = graphInfoSrc->numGraphInputs;
    }
    graphInfoDst->outputTensors = nullptr;
    graphInfoDst->numOutputTensors = 0;
    if (graphInfoSrc->graphOutputs) {
        if (!copyTensorsInfo(graphInfoSrc->graphOutputs,
                             graphInfoDst->outputTensors,
                             graphInfoSrc->numGraphOutputs)) {
            return false;
        }
        graphInfoDst->numOutputTensors = graphInfoSrc->numGraphOutputs;
    }
    return true;
}

bool copyGraphsInfoV3(const QnnSystemContext_GraphInfoV3_t *graphInfoSrc,
                      GraphInfo_t *graphInfoDst) {
    graphInfoDst->graphName = nullptr;
    if (graphInfoSrc->graphName) {
        graphInfoDst->graphName =
            strndup(graphInfoSrc->graphName, strlen(graphInfoSrc->graphName));
    }
    graphInfoDst->inputTensors = nullptr;
    graphInfoDst->numInputTensors = 0;
    if (graphInfoSrc->graphInputs) {
        if (!copyTensorsInfo(
                graphInfoSrc->graphInputs, graphInfoDst->inputTensors, graphInfoSrc->numGraphInputs)) {
            return false;
        }
        graphInfoDst->numInputTensors = graphInfoSrc->numGraphInputs;
    }
    graphInfoDst->outputTensors = nullptr;
    graphInfoDst->numOutputTensors = 0;
    if (graphInfoSrc->graphOutputs) {
        if (!copyTensorsInfo(graphInfoSrc->graphOutputs,
                             graphInfoDst->outputTensors,
                             graphInfoSrc->numGraphOutputs)) {
            return false;
        }
        graphInfoDst->numOutputTensors = graphInfoSrc->numGraphOutputs;
    }
    return true;
}

bool copyGraphsInfo(const QnnSystemContext_GraphInfo_t *graphsInput,
                    const uint32_t numGraphs,
                    GraphInfo_t **&graphsInfo) {
    if (!graphsInput) {
        MLLM_LOG_ERROR("Received nullptr for graphsInput.");
        return false;
    }
    auto returnStatus = true;
    graphsInfo =
        (GraphInfo_t **)calloc(numGraphs, sizeof(GraphInfo_t *));
    GraphInfo_t *graphInfoArr =
        (GraphInfo_t *)calloc(numGraphs, sizeof(GraphInfo_t));
    if (nullptr == graphsInfo || nullptr == graphInfoArr) {
        MLLM_LOG_ERROR("Failure to allocate memory for *graphInfo");
        returnStatus = false;
    }
    if (true == returnStatus) {
        for (size_t gIdx = 0; gIdx < numGraphs; gIdx++) {
            if (graphsInput[gIdx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
                copyGraphsInfoV1(&graphsInput[gIdx].graphInfoV1, &graphInfoArr[gIdx]);
            } else if (graphsInput[gIdx].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
                copyGraphsInfoV3(&graphsInput[gIdx].graphInfoV3, &graphInfoArr[gIdx]);
            }
            graphsInfo[gIdx] = graphInfoArr + gIdx;
        }
    }
    if (true != returnStatus) {
        MLLM_LOG_ERROR("Received an ERROR during extractGraphsInfo. Freeing resources.");
        if (graphsInfo) {
            for (uint32_t gIdx = 0; gIdx < numGraphs; gIdx++) {
                if (graphsInfo[gIdx]) {
                    if (nullptr != graphsInfo[gIdx]->graphName) {
                        free(graphsInfo[gIdx]->graphName);
                        graphsInfo[gIdx]->graphName = nullptr;
                    }
                    freeQnnTensors(graphsInfo[gIdx]->inputTensors,
                                   graphsInfo[gIdx]->numInputTensors);
                    freeQnnTensors(graphsInfo[gIdx]->outputTensors,
                                   graphsInfo[gIdx]->numOutputTensors);
                }
            }
            free(*graphsInfo);
        }
        free(graphsInfo);
        graphsInfo = nullptr;
    }
    return true;
}

bool copyTensorsInfo(const Qnn_Tensor_t *tensorsInfoSrc,
                     Qnn_Tensor_t *&tensorWrappers,
                     uint32_t tensorsCount) {
    auto returnStatus = true;
    tensorWrappers = (Qnn_Tensor_t *)calloc(tensorsCount, sizeof(Qnn_Tensor_t));
    if (nullptr == tensorWrappers) {
        MLLM_LOG_ERROR("Failed to allocate memory for tensorWrappers.");
        return false;
    }
    for (size_t tIdx = 0; tIdx < tensorsCount; tIdx++) {
        tensorWrappers[tIdx] = QNN_TENSOR_INIT;
        deepCopyQnnTensorInfo(&tensorWrappers[tIdx], &tensorsInfoSrc[tIdx]);
    }
    return true;
}

bool deepCopyQnnTensorInfo(Qnn_Tensor_t *dst, const Qnn_Tensor_t *src) {
    if (nullptr == dst || nullptr == src) {
        MLLM_LOG_ERROR("Received nullptr");
        return false;
    }
    // set tensor.version before using QNN_TENSOR_SET macros, as they require the version to be set
    // to correctly assign values
    dst->version = src->version;
    const char *tensorName = QNN_TENSOR_GET_NAME(src);
    if (!tensorName) {
        QNN_TENSOR_SET_NAME(dst, nullptr);
    } else {
        QNN_TENSOR_SET_NAME(dst, ::strndup(tensorName, strlen(tensorName)));
    }
    QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
    QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
    QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
    QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));
    Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
    qParams.encodingDefinition = QNN_TENSOR_GET_QUANT_PARAMS(src).encodingDefinition;
    qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
        qParams.scaleOffsetEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).scaleOffsetEncoding;
    } else if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
        qParams.axisScaleOffsetEncoding.axis =
            QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.axis;
        qParams.axisScaleOffsetEncoding.numScaleOffsets =
            QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
        if (QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets > 0) {
            qParams.axisScaleOffsetEncoding.scaleOffset = (Qnn_ScaleOffset_t *)malloc(
                QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets * sizeof(Qnn_ScaleOffset_t));
            if (qParams.axisScaleOffsetEncoding.scaleOffset) {
                for (size_t idx = 0;
                     idx < QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
                     idx++) {
                    qParams.axisScaleOffsetEncoding.scaleOffset[idx].scale =
                        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].scale;
                    qParams.axisScaleOffsetEncoding.scaleOffset[idx].offset =
                        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].offset;
                }
            }
        }
    }
    QNN_TENSOR_SET_QUANT_PARAMS(dst, qParams);
    QNN_TENSOR_SET_RANK(dst, QNN_TENSOR_GET_RANK(src));
    QNN_TENSOR_SET_DIMENSIONS(dst, nullptr);

    auto memscpy = [](void *dst, size_t dstSize, const void *src, size_t copySize) -> size_t {
        if (!dst || !src || !dstSize || !copySize) return 0;

        size_t minSize = dstSize < copySize ? dstSize : copySize;

        memcpy(dst, src, minSize);

        return minSize;
    };
    if (QNN_TENSOR_GET_RANK(src) > 0) {
        QNN_TENSOR_SET_DIMENSIONS(dst, (uint32_t *)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t)));
        if (QNN_TENSOR_GET_DIMENSIONS(dst)) {
            memscpy(QNN_TENSOR_GET_DIMENSIONS(dst),
                    QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t),
                    QNN_TENSOR_GET_DIMENSIONS(src),
                    QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t));
        }
        if (QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src)) {
            QNN_TENSOR_SET_IS_DYNAMIC_DIMENSIONS(
                dst, (uint8_t *)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint8_t)));
            memscpy(QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(dst),
                    QNN_TENSOR_GET_RANK(src) * sizeof(uint8_t),
                    QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src),
                    QNN_TENSOR_GET_RANK(src) * sizeof(uint8_t));
        }
    }
    QNN_TENSOR_SET_SPARSE_PARAMS(dst, QNN_TENSOR_GET_SPARSE_PARAMS(src));
    return true;
}

bool freeQnnTensor(Qnn_Tensor_t &tensor) {
    // free all pointer allocations in struct
    free((void *)QNN_TENSOR_GET_NAME(tensor));
    free(QNN_TENSOR_GET_DIMENSIONS(tensor));
    if (QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(tensor)) {
        free(QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(tensor));
    }
    auto quant = QNN_TENSOR_GET_QUANT_PARAMS(tensor);
    auto encoding = quant.quantizationEncoding;
    if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        if (quant.axisScaleOffsetEncoding.scaleOffset != nullptr) {
            free(quant.axisScaleOffsetEncoding.scaleOffset);
        }
    }
    return true;
}

bool freeQnnTensors(Qnn_Tensor_t *&tensors,
                    uint32_t numTensors) {
    // free all pointer allocations in struct
    for (size_t i = 0; i < numTensors; i++) {
        freeQnnTensor(tensors[i]);
    }
    free(tensors);
    return true;
}

} // namespace mllm