//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <stdlib.h>

#include "QnnTypeMacros.hpp"
#include "QnnWrapperUtils.hpp"

qnn_wrapper_api::ModelError_t qnn_wrapper_api::freeQnnTensor(Qnn_Tensor_t &tensor) {
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
    return MODEL_NO_ERROR;
}

qnn_wrapper_api::ModelError_t qnn_wrapper_api::freeQnnTensors(Qnn_Tensor_t *&tensors,
                                                              uint32_t numTensors) {
    // free all pointer allocations in struct
    for (size_t i = 0; i < numTensors; i++) {
        freeQnnTensor(tensors[i]);
    }
    free(tensors);
    return MODEL_NO_ERROR;
}

qnn_wrapper_api::ModelError_t qnn_wrapper_api::freeGraphsInfo(GraphInfoPtr_t **graphsInfo,
                                                              uint32_t numGraphs) {
    if (graphsInfo == nullptr || *graphsInfo == nullptr) {
        return MODEL_TENSOR_ERROR;
    }
    for (uint32_t i = 0; i < numGraphs; i++) {
        free((*graphsInfo)[i]->graphName);
        freeQnnTensors((*graphsInfo)[i]->inputTensors, (*graphsInfo)[i]->numInputTensors);
        freeQnnTensors((*graphsInfo)[i]->outputTensors, (*graphsInfo)[i]->numOutputTensors);
    }
    free(**graphsInfo);
    free(*graphsInfo);
    *graphsInfo = nullptr;
    return MODEL_NO_ERROR;
}
