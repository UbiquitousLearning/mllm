//==============================================================================
//
//  Copyright (c) 2019-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <cstdlib>
#include <cstring>
#include <string>

#include "QnnModelPal.hpp"
#include "QnnTypeMacros.hpp"
#include "QnnWrapperUtils.hpp"

namespace qnn_wrapper_api {
size_t memscpy(void *dst, size_t dstSize, const void *src, size_t copySize) {
  if (!dst || !src || !dstSize || !copySize) return 0;

  size_t minSize = dstSize < copySize ? dstSize : copySize;

  memcpy(dst, src, minSize);

  return minSize;
}

ModelError_t getQnnGraphConfigFromInfo(const char *graphName,
                                       const GraphConfigInfo_t **graphsConfigInfo,
                                       const uint32_t numGraphsConfigInfo,
                                       const QnnGraph_Config_t **&graphConfigs) {
  if (!graphsConfigInfo || numGraphsConfigInfo == 0) {
    PRINT_DEBUG("getQnnGraphConfigFromInfo() no custom configs passed for graph:%s.\n", graphName);
    return MODEL_NO_ERROR;
  }

  size_t found = 0;

  for (uint32_t i = 0; i < numGraphsConfigInfo; i++) {
    if (!graphsConfigInfo[i]) {
      PRINT_ERROR(
          "getQnnGraphConfigFromInfo() lookup error while trying to query graphName:%s. "
          "numGraphsConfigInfo > num of element in graphsConfigInfo\n",
          graphName);
      return MODEL_INVALID_ARGUMENT_ERROR;
    }
    if (strcmp(graphsConfigInfo[i]->graphName, graphName) == 0) {
      graphConfigs = graphsConfigInfo[i]->graphConfigs;
      found++;
    }
  }

  if (!found) {
    PRINT_ERROR(
        "getQnnGraphConfigFromInfo() unable to find graphName:%s in provided "
        "graphsConfigInfo object.\n",
        graphName);
    return MODEL_INVALID_ARGUMENT_ERROR;
  } else if (found > 1) {
    PRINT_ERROR(
        "getQnnGraphConfigFromInfo() duplicate GraphConfigInfo entries found with "
        "graphName:%s.\n",
        graphName);
    return MODEL_INVALID_ARGUMENT_ERROR;
  } else {
    return MODEL_NO_ERROR;
  }
}

ModelError_t deepCopyQnnTensors(Qnn_Tensor_t &src, Qnn_Tensor_t &dst) {
  ModelError_t err;
  VALIDATE_TENSOR_VERSION(src, err);

  dst.version = src.version;
  QNN_TENSOR_SET_NAME(
      dst, strnDup(QNN_TENSOR_GET_NAME(src), std::string(QNN_TENSOR_GET_NAME(src)).size()));
  if (QNN_TENSOR_GET_NAME(dst) == nullptr) {
    return MODEL_TENSOR_ERROR;
  }
  QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
  QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
  QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
  QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));
  QNN_TENSOR_SET_MEM_TYPE(dst, QNN_TENSOR_GET_MEM_TYPE(src));

  // Only metadata (i.e. non-static data) is copied from source to destination. The union still
  // must be initialized so that the clientBuf/memHandle do not contain garbage data
  if (QNN_TENSOR_GET_MEM_TYPE(src) == QNN_TENSORMEMTYPE_RAW) {
    Qnn_ClientBuffer_t clientBuf = {nullptr, 0};
    QNN_TENSOR_SET_CLIENT_BUF(dst, clientBuf);
  } else if (QNN_TENSOR_GET_MEM_TYPE(src) == QNN_TENSORMEMTYPE_MEMHANDLE) {
    QNN_TENSOR_SET_MEM_HANDLE(dst, nullptr);
  } else {
    return MODEL_TENSOR_ERROR;
  }

  Qnn_QuantizeParams_t srcQParam      = QNN_TENSOR_GET_QUANT_PARAMS(src);
  Qnn_QuantizationEncoding_t encoding = srcQParam.quantizationEncoding;
  if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    // need to allocate and copy memory for scaleOffset as it is a pointer array
    Qnn_QuantizeParams_t srcQParamCpy      = srcQParam;
    Qnn_AxisScaleOffset_t &axisScaleOffset = srcQParamCpy.axisScaleOffsetEncoding;
    Qnn_ScaleOffset_t **scaleOffset        = &axisScaleOffset.scaleOffset;
    size_t scaleOffsetSize = axisScaleOffset.numScaleOffsets * sizeof(Qnn_ScaleOffset_t);
    *scaleOffset           = (Qnn_ScaleOffset_t *)malloc(scaleOffsetSize);
    memscpy(*scaleOffset,
            scaleOffsetSize,
            srcQParam.axisScaleOffsetEncoding.scaleOffset,
            scaleOffsetSize);
    QNN_TENSOR_SET_QUANT_PARAMS(dst, srcQParamCpy);
  } else if (encoding == QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET) {
    // need to allocate and copy memory for scaleOffset as it is a pointer array
    Qnn_QuantizeParams_t srcQParamCpy          = srcQParam;
    Qnn_BwAxisScaleOffset_t &bwAxisScaleOffset = srcQParamCpy.bwAxisScaleOffsetEncoding;
    size_t scaleSize                           = bwAxisScaleOffset.numElements * sizeof(float);
    float **scales                             = &bwAxisScaleOffset.scales;
    int32_t **offsets                          = &bwAxisScaleOffset.offsets;
    *scales                                    = (float *)malloc(scaleSize);
    memscpy(*scales, scaleSize, srcQParam.bwAxisScaleOffsetEncoding.scales, scaleSize);

    // Only copy offsets if present, nullptr implies all offsets are 0
    if (bwAxisScaleOffset.offsets != nullptr) {
      size_t offsetSize = bwAxisScaleOffset.numElements * sizeof(int32_t);
      *offsets          = (int32_t *)malloc(offsetSize);
      memscpy(*offsets, offsetSize, srcQParam.bwAxisScaleOffsetEncoding.offsets, offsetSize);
    }
    QNN_TENSOR_SET_QUANT_PARAMS(dst, srcQParamCpy);
  } else {
    QNN_TENSOR_SET_QUANT_PARAMS(dst, srcQParam);
  }

  // need to allocate and copy memory for all the pointer members
  uint32_t rank = QNN_TENSOR_GET_RANK(src);
  QNN_TENSOR_SET_RANK(dst, rank);
  size_t dimSize       = rank * sizeof(uint32_t);
  uint32_t *dimensions = (uint32_t *)malloc(dimSize);
  if (dimensions == nullptr) {
    PRINT_ERROR("deepCopyQnnTensors() Allocation error while copying tensor %s",
                QNN_TENSOR_GET_NAME(src));
    return MODEL_TENSOR_ERROR;
  }
  memscpy(dimensions, dimSize, QNN_TENSOR_GET_DIMENSIONS(src), dimSize);
  QNN_TENSOR_SET_DIMENSIONS(dst, dimensions);

  return err;
}

ModelError_t freeQnnTensor(Qnn_Tensor_t &tensor) {
  ModelError_t err;
  VALIDATE_TENSOR_VERSION(tensor, err);

  // free all pointer allocations in struct
  free((void *)QNN_TENSOR_GET_NAME(tensor));
  free(QNN_TENSOR_GET_DIMENSIONS(tensor));

  return MODEL_NO_ERROR;
}

ModelError_t freeQnnTensors(Qnn_Tensor_t *&tensors, uint32_t numTensors) {
  // free all pointer allocations in struct
  for (size_t i = 0; i < numTensors; i++) {
    freeQnnTensor(tensors[i]);
  }
  free(tensors);

  return MODEL_NO_ERROR;
}

std::string getModelErrorName(ModelError_t modelError) {
  switch (modelError) {
    case MODEL_NO_ERROR:
      return "MODEL_NO_ERROR";
    case MODEL_TENSOR_ERROR:
      return "MODEL_TENSOR_ERROR";
    case MODEL_PARAMS_ERROR:
      return "MODEL_PARAMS_ERROR";
    case MODEL_NODES_ERROR:
      return "MODEL_NODES_ERROR";
    case MODEL_GRAPH_ERROR:
      return "MODEL_GRAPH_ERROR";
    case MODEL_CONTEXT_ERROR:
      return "MODEL_CONTEXT_ERROR";
    case MODEL_GENERATION_ERROR:
      return "MODEL_GENERATION_ERROR";
    case MODEL_SETUP_ERROR:
      return "MODEL_SETUP_ERROR";
    case MODEL_UNKNOWN_ERROR:
      return "MODEL_UNKNOWN_ERROR";
    case MODEL_INVALID_ARGUMENT_ERROR:
      return "MODEL_INVALID_ARGUMENT_ERROR";
    case MODEL_FILE_ERROR:
      return "MODEL_FILE_ERROR";
    default:
      return "INVALID_ERROR_CODE";
  }
}

}  // namespace qnn_wrapper_api
