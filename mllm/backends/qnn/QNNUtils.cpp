#include "QNNUtils.hpp"
#include "QnnTypeMacros.hpp"
#include "mllm/utils/Log.hpp"
#include <cstdint>
#include <memory>
#include <dlfcn.h>

namespace mllm::qnn {

// --------------- Begin of QNN symbols loading ---------------

QnnInterfaceGetProvidersFn_t QnnInterface_getProviders = nullptr;

bool loadQNNSymbol() {
  MLLM_INFO("QNN Backend Lib: libQnnHtp.so");
  void* qnnLibHandle = nullptr;
  qnnLibHandle = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
  const char* errorOpen = dlerror();
  if (!qnnLibHandle) {
    MLLM_ERROR("Failed to open QNN libs.");
    return false;
  }

  QnnInterface_getProviders = (QnnInterfaceGetProvidersFn_t)dlsym(qnnLibHandle, "QnnInterface_getProviders");
  const char* errorSym = dlerror();
  if (!QnnInterface_getProviders) {
    MLLM_ERROR("Failed to load symbol <QnnInterface_getProviders>. dlerror returns %s.\n", errorSym);
    dlclose(qnnLibHandle);
    return false;
  }

  return true;
}

QnnSystemInterfaceGetProvidersFn_t QnnSystemInterface_getProviders = nullptr;

bool loadQNNSystemSymbol() {
  void* systemLibraryHandle = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
  const char* errorOpen = dlerror();
  if (!systemLibraryHandle) {
    MLLM_ERROR("Failed to open QNN System libs.");
    return false;
  }

  QnnSystemInterface_getProviders =
      (QnnSystemInterfaceGetProvidersFn_t)dlsym(systemLibraryHandle, "QnnSystemInterface_getProviders");
  const char* errorSym = dlerror();
  if (!QnnSystemInterface_getProviders) {
    MLLM_ERROR("Failed to load symbol <QnnSystemInterface_getProviders>. dlerror returns %s.\n", errorSym);
    dlclose(systemLibraryHandle);
    return false;
  }

  return true;
}

// --------------- End of QNN symbols loading ---------------

bool copyMetadataToGraphsInfo(const QnnSystemContext_BinaryInfo_t* binaryInfo, GraphInfo_t**& graphsInfo,
                              uint32_t& graphsCount) {
  if (nullptr == binaryInfo) {
    MLLM_ERROR("binaryInfo is nullptr.");
    return false;
  }
  graphsCount = 0;
  if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    if (binaryInfo->contextBinaryInfoV1.graphs) {
      if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV1.graphs, binaryInfo->contextBinaryInfoV1.numGraphs, graphsInfo)) {
        MLLM_ERROR("Failed while copying graphs Info.");
        return false;
      }
      graphsCount = binaryInfo->contextBinaryInfoV1.numGraphs;
      return true;
    }
  } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    if (binaryInfo->contextBinaryInfoV2.graphs) {
      if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV2.graphs, binaryInfo->contextBinaryInfoV2.numGraphs, graphsInfo)) {
        MLLM_ERROR("Failed while copying graphs Info.");
        return false;
      }
      graphsCount = binaryInfo->contextBinaryInfoV2.numGraphs;
      return true;
    }
  } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    if (binaryInfo->contextBinaryInfoV3.graphs) {
      if (!copyGraphsInfo(binaryInfo->contextBinaryInfoV3.graphs, binaryInfo->contextBinaryInfoV3.numGraphs, graphsInfo)) {
        MLLM_ERROR("Failed while copying graphs Info.");
        return false;
      }
      graphsCount = binaryInfo->contextBinaryInfoV3.numGraphs;
      return true;
    }
  }
  MLLM_ERROR("Unrecognized system context binary info version.");
  return false;
}

bool copyGraphsInfoV1(const QnnSystemContext_GraphInfoV1_t* graphInfoSrc, GraphInfo_t* graphInfoDst) {
  graphInfoDst->graphName = nullptr;
  if (graphInfoSrc->graphName) { graphInfoDst->graphName = strndup(graphInfoSrc->graphName, strlen(graphInfoSrc->graphName)); }
  graphInfoDst->inputTensors = nullptr;
  graphInfoDst->numInputTensors = 0;
  if (graphInfoSrc->graphInputs) {
    if (!copyTensorsInfo(graphInfoSrc->graphInputs, graphInfoDst->inputTensors, graphInfoSrc->numGraphInputs)) { return false; }
    graphInfoDst->numInputTensors = graphInfoSrc->numGraphInputs;
  }
  graphInfoDst->outputTensors = nullptr;
  graphInfoDst->numOutputTensors = 0;
  if (graphInfoSrc->graphOutputs) {
    if (!copyTensorsInfo(graphInfoSrc->graphOutputs, graphInfoDst->outputTensors, graphInfoSrc->numGraphOutputs)) {
      return false;
    }
    graphInfoDst->numOutputTensors = graphInfoSrc->numGraphOutputs;
  }
  return true;
}

bool copyGraphsInfoV3(const QnnSystemContext_GraphInfoV3_t* graphInfoSrc, GraphInfo_t* graphInfoDst) {
  graphInfoDst->graphName = nullptr;
  if (graphInfoSrc->graphName) { graphInfoDst->graphName = strndup(graphInfoSrc->graphName, strlen(graphInfoSrc->graphName)); }
  graphInfoDst->inputTensors = nullptr;
  graphInfoDst->numInputTensors = 0;
  if (graphInfoSrc->graphInputs) {
    if (!copyTensorsInfo(graphInfoSrc->graphInputs, graphInfoDst->inputTensors, graphInfoSrc->numGraphInputs)) { return false; }
    graphInfoDst->numInputTensors = graphInfoSrc->numGraphInputs;
  }
  graphInfoDst->outputTensors = nullptr;
  graphInfoDst->numOutputTensors = 0;
  if (graphInfoSrc->graphOutputs) {
    if (!copyTensorsInfo(graphInfoSrc->graphOutputs, graphInfoDst->outputTensors, graphInfoSrc->numGraphOutputs)) {
      return false;
    }
    graphInfoDst->numOutputTensors = graphInfoSrc->numGraphOutputs;
  }
  return true;
}

bool copyGraphsInfo(const QnnSystemContext_GraphInfo_t* graphsInput, const uint32_t numGraphs, GraphInfo_t**& graphsInfo) {
  if (!graphsInput) {
    MLLM_ERROR("Received nullptr for graphsInput.");
    return false;
  }
  auto returnStatus = true;
  graphsInfo = (GraphInfo_t**)calloc(numGraphs, sizeof(GraphInfo_t*));
  GraphInfo_t* graphInfoArr = (GraphInfo_t*)calloc(numGraphs, sizeof(GraphInfo_t));
  if (nullptr == graphsInfo || nullptr == graphInfoArr) {
    MLLM_ERROR("Failure to allocate memory for *graphInfo");
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
    MLLM_ERROR("Received an ERROR during extractGraphsInfo. Freeing resources.");
    if (graphsInfo) {
      for (uint32_t gIdx = 0; gIdx < numGraphs; gIdx++) {
        if (graphsInfo[gIdx]) {
          if (nullptr != graphsInfo[gIdx]->graphName) {
            free(graphsInfo[gIdx]->graphName);
            graphsInfo[gIdx]->graphName = nullptr;
          }
          freeQnnTensors(graphsInfo[gIdx]->inputTensors, graphsInfo[gIdx]->numInputTensors);
          freeQnnTensors(graphsInfo[gIdx]->outputTensors, graphsInfo[gIdx]->numOutputTensors);
        }
      }
      free(*graphsInfo);
    }
    free(graphsInfo);
    graphsInfo = nullptr;
  }
  return true;
}

bool copyTensorsInfo(const Qnn_Tensor_t* tensorsInfoSrc, Qnn_Tensor_t*& tensorWrappers, uint32_t tensorsCount) {
  auto returnStatus = true;
  tensorWrappers = (Qnn_Tensor_t*)calloc(tensorsCount, sizeof(Qnn_Tensor_t));
  if (nullptr == tensorWrappers) {
    MLLM_ERROR("Failed to allocate memory for tensorWrappers.");
    return false;
  }
  for (size_t tIdx = 0; tIdx < tensorsCount; tIdx++) {
    tensorWrappers[tIdx] = QNN_TENSOR_INIT;
    deepCopyQnnTensorInfo(&tensorWrappers[tIdx], &tensorsInfoSrc[tIdx]);
  }
  return true;
}

bool deepCopyQnnTensorInfo(Qnn_Tensor_t* dst, const Qnn_Tensor_t* src) {
  if (nullptr == dst || nullptr == src) {
    MLLM_ERROR("Received nullptr");
    return false;
  }
  // set tensor.version before using QNN_TENSOR_SET macros, as they require the version to be set
  // to correctly assign values
  dst->version = src->version;
  const char* tensorName = src->v2.name;
  if (!tensorName) {
    QNN_TENSOR_SET_NAME(dst, nullptr);
  } else {
    QNN_TENSOR_SET_NAME(dst, ::strndup(tensorName, strlen(tensorName)));
  }
  dst->v2.id = src->v2.id;
  dst->v2.type = src->v2.type;
  dst->v2.dataFormat = src->v2.dataFormat;
  dst->v2.dataType = src->v2.dataType;
  Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
  qParams.encodingDefinition = src->v2.quantizeParams.encodingDefinition;
  qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  if (src->v2.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    qParams.quantizationEncoding = src->v2.quantizeParams.quantizationEncoding;
    qParams.scaleOffsetEncoding = src->v2.quantizeParams.scaleOffsetEncoding;
  } else if (src->v2.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    qParams.quantizationEncoding = src->v2.quantizeParams.quantizationEncoding;
    qParams.axisScaleOffsetEncoding.axis = src->v2.quantizeParams.axisScaleOffsetEncoding.axis;
    qParams.axisScaleOffsetEncoding.numScaleOffsets = src->v2.quantizeParams.axisScaleOffsetEncoding.numScaleOffsets;
    if (src->v2.quantizeParams.axisScaleOffsetEncoding.numScaleOffsets > 0) {
      qParams.axisScaleOffsetEncoding.scaleOffset = (Qnn_ScaleOffset_t*)malloc(
          src->v2.quantizeParams.axisScaleOffsetEncoding.numScaleOffsets * sizeof(Qnn_ScaleOffset_t));
      if (qParams.axisScaleOffsetEncoding.scaleOffset) {
        for (size_t idx = 0; idx < src->v2.quantizeParams.axisScaleOffsetEncoding.numScaleOffsets; idx++) {
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].scale =
              src->v2.quantizeParams.axisScaleOffsetEncoding.scaleOffset[idx].scale;
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].offset =
              src->v2.quantizeParams.axisScaleOffsetEncoding.scaleOffset[idx].offset;
        }
      }
    }
  }

  dst->v2.quantizeParams = qParams;
  dst->v2.rank = src->v2.rank;
  dst->v2.dimensions = nullptr;

  auto memscpy = [](void* dst, size_t dstSize, const void* src, size_t copySize) -> size_t {
    if (!dst || !src || !dstSize || !copySize) return 0;

    size_t minSize = dstSize < copySize ? dstSize : copySize;

    memcpy(dst, src, minSize);

    return minSize;
  };
  if (src->v2.rank > 0) {
    QNN_TENSOR_SET_DIMENSIONS(dst, (uint32_t*)malloc(src->v2.rank * sizeof(uint32_t)));
    if (dst->v2.dimensions) {
      memscpy(dst->v2.dimensions, src->v2.rank * sizeof(uint32_t), src->v2.dimensions, src->v2.rank * sizeof(uint32_t));
    }
    if (src->version == QNN_TENSOR_VERSION_2 && src->v2.isDynamicDimensions) {
      if (dst->version == QNN_TENSOR_VERSION_2) {
        dst->v2.isDynamicDimensions = (uint8_t*)malloc(src->v2.rank * sizeof(uint8_t));
        memscpy(dst->v2.isDynamicDimensions, src->v2.rank * sizeof(uint8_t), src->v2.isDynamicDimensions,
                src->v2.rank * sizeof(uint8_t));
      }
    }
  }

  if (dst->version == QNN_TENSOR_VERSION_2 && src->version == QNN_TENSOR_VERSION_2) {
    dst->v2.sparseParams = src->v2.sparseParams;
  }

  return true;
}

bool freeQnnTensor(Qnn_Tensor_t& tensor) {
  // free all pointer allocations in struct
  free((void*)tensor.v2.name);
  free(tensor.v2.dimensions);
  if (tensor.version == QNN_TENSOR_VERSION_2 && tensor.v2.isDynamicDimensions) { free(tensor.v2.isDynamicDimensions); }
  auto quant = tensor.v2.quantizeParams;
  auto encoding = quant.quantizationEncoding;
  if (encoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    if (quant.axisScaleOffsetEncoding.scaleOffset != nullptr) { free(quant.axisScaleOffsetEncoding.scaleOffset); }
  }
  return true;
}

bool freeQnnTensors(Qnn_Tensor_t*& tensors, uint32_t numTensors) {
  // free all pointer allocations in struct
  for (size_t i = 0; i < numTensors; i++) { freeQnnTensor(tensors[i]); }
  free(tensors);
  return true;
}

}  // namespace mllm::qnn