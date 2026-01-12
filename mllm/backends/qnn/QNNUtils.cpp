#include <cstdint>
#include <memory>
#include <dlfcn.h>
#include <cstring>

#include "QNNUtils.hpp"
#include "QnnTypes.h"

#include "mllm/backends/qnn/QNNAllocator.hpp"
#include "mllm/backends/qnn/QNNTypeMacros.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/utils/Log.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"

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
    MLLM_ERROR("Failed to load symbol <QnnInterface_getProviders>. dlerror returns {}.", errorSym);
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
    MLLM_ERROR("Failed to load symbol <QnnSystemInterface_getProviders>. dlerror returns {}.", errorSym);
    dlclose(systemLibraryHandle);
    return false;
  }

  return true;
}

// --------------- End of QNN symbols loading ---------------

// --------------- QNN Graph Info Copying methods ---------------
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

// --------------- QNN Wrapper ---------------

Qnn_DataType_t mllmDataTypeToQnnDataType(DataTypes dtype) {
  Qnn_DataType_t ret = QNN_DATATYPE_UNDEFINED;
  switch (dtype) {
    case kInt8: {
      ret = QNN_DATATYPE_SFIXED_POINT_8;
      break;
    }
    case kInt16: {
      ret = QNN_DATATYPE_UFIXED_POINT_16;
      break;
    }
    case kInt32: {
      ret = QNN_DATATYPE_INT_32;
      break;
    }
    case kInt64: {
      ret = QNN_DATATYPE_INT_64;
      break;
    }
    case kUInt8: {
      ret = QNN_DATATYPE_UFIXED_POINT_8;
      break;
    }
    case kUInt16: {
      ret = QNN_DATATYPE_UFIXED_POINT_16;
      break;
    }
    case kUInt32: {
      ret = QNN_DATATYPE_UINT_32;
      break;
    }
    case kUInt64: {
      ret = QNN_DATATYPE_UINT_64;
      break;
    }
    case kFloat16: {
      ret = QNN_DATATYPE_FLOAT_16;
      break;
    }
    case kFloat32: {
      ret = QNN_DATATYPE_FLOAT_32;
      break;
    }
    // case kBFloat16: {
    //   ret = QNN_DATATYPE_BFLOAT_16;
    //   break;
    // }
    // FIXME: Maybe error here.
    case kInt4: {
      ret = QNN_DATATYPE_SFIXED_POINT_4;
      break;
    }
    case kUInt4: {
      ret = QNN_DATATYPE_UFIXED_POINT_4;
      break;
    }
    case kInt8PerTensorSym:
    case kInt8PerTensorAsy:
    case kInt8PerChannelAsy:
    case kInt8PerChannelSym: {
      ret = QNN_DATATYPE_SFIXED_POINT_8;
      break;
    }
    case kUInt8PerTensorSym:
    case kUInt8PerTensorAsy:
    case kUInt8PerChannelAsy:
    case kUInt8PerChannelSym: {
      ret = QNN_DATATYPE_UFIXED_POINT_8;
      break;
    }
    case kInt16PerTensorSym:
    case kInt16PerTensorAsy:
    case kInt16PerChannelSym:
    case kInt16PerChannelAsy: {
      ret = QNN_DATATYPE_SFIXED_POINT_16;
      break;
    }
    case kUInt16PerTensorSym:
    case kUInt16PerTensorAsy:
    case kUInt16PerChannelSym:
    case kUInt16PerChannelAsy: {
      ret = QNN_DATATYPE_UFIXED_POINT_16;
      break;
    }
    default: {
      MLLM_ERROR("Can't parse datatype: {}", nameOfType(dtype));
      ret = QNN_DATATYPE_UNDEFINED;
    }
  }
  return ret;
}

size_t qnnDataTypeToSize(Qnn_DataType_t dtype) {
  switch (dtype) {
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_BOOL_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8: return 1;

    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16: return 2;

    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32: return 4;

    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64: return 8;

    default:
      MLLM_ERROR("qnnDataTypeToSize: unsupported Qnn_DataType_t {}", static_cast<int>(dtype));
      MLLM_RT_ASSERT(false);
      return 0;
  }
}

QNNTensorWrapper::QNNTensorWrapper(const std::string& name, Qnn_TensorType_t type, Qnn_DataType_t dataType,
                                   const std::vector<uint32_t>& dimensions, Qnn_QuantizeParams_t quantize) {
  name_ = name;
  dimensions_ = dimensions;

  // init QNN tensor
  // QNN tensor v2 is binary compatible with v1, so we can use v2 directly
  qnnTensor_.version = QNN_TENSOR_VERSION_2;

  qnnTensor_.v2 = QNN_TENSOR_V2_INIT;

  qnnTensor_.v2.name = name_.c_str();
  qnnTensor_.v2.type = type;
  qnnTensor_.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_DENSE;
  qnnTensor_.v2.dataType = dataType;
  qnnTensor_.v2.quantizeParams = quantize;
  qnnTensor_.v2.rank = dimensions_.size();
  qnnTensor_.v2.dimensions = dimensions_.data();
  qnnTensor_.v2.memType = QNN_TENSORMEMTYPE_RAW;
  qnnTensor_.v2.clientBuf.data = nullptr;
  qnnTensor_.v2.clientBuf.dataSize = 0;
}

std::shared_ptr<QNNTensorWrapper> QNNTensorWrapper::create(const std::string& name, Qnn_TensorType_t type, const Tensor& tensor,
                                                           Qnn_QuantizeParams_t quantize) {
  // in this case, the tensor may be a placeholder(input/output except for graph IO)
  // it will be allocated to QNN shared buffer via QNNTensorWrapper::alloc() later
  MLLM_RT_ASSERT(!name.empty());
  // in AOT case, the tensor is all on CPU (TODO: handle this)
  // if (type != QNN_TENSOR_TYPE_STATIC) { MLLM_RT_ASSERT(tensor.device() == kQNN); }

  Qnn_DataType_t dataType = mllmDataTypeToQnnDataType(tensor.dtype());

  std::vector<uint32_t> dimensions(tensor.rank());
  for (int i = 0; i < tensor.rank(); i++) { dimensions[i] = tensor.shape()[i]; }

  auto tensorWrapper = std::make_shared<QNNTensorWrapper>(name, type, dataType, dimensions, quantize);

  tensorWrapper->dataContainer_ = tensor;

  return tensorWrapper;
}

std::shared_ptr<QNNTensorWrapper> QNNTensorWrapper::createStaticTensor(const std::string& name, const Tensor& tensor,
                                                                       Qnn_QuantizeParams_t quantize) {
  MLLM_RT_ASSERT(!name.empty() && tensor.rank() > 0 && !tensor.isNil());

  std::shared_ptr<QNNTensorWrapper> tensorWrapper = QNNTensorWrapper::create(name, QNN_TENSOR_TYPE_STATIC, tensor, quantize);

  tensorWrapper->isAlloc_ = true;
  tensorWrapper->registeredPtr_ = tensor.ptr<void>();

  uint32_t numElement = tensor.bytes();
  Qnn_ClientBuffer_t clientBuffer = {.data = tensor.ptr<void>(), .dataSize = numElement};

  QNN_TENSOR_SET_CLIENT_BUF(tensorWrapper->qnnTensor_, clientBuffer);
  return tensorWrapper;
}

void QNNTensorWrapper::alloc() {
  if (isAlloc_) {
    MLLM_WARN("Tensor {} has already been allocated.", name_);
    return;
  }
  MLLM_RT_ASSERT(dataContainer_.device() == kQNN);

  // if storage is not allocated, allocate it
  // or, register the existing storage to QNN(passing allocated input to QNN)
  if (!dataContainer_.impl()->ptr<void>()) { dataContainer_.alloc(); }

  std::static_pointer_cast<QNNAllocator>(Context::instance().getBackend(kQNN)->allocator())
      ->registerQnnTensorToSharedBuffer(dataContainer_.ptr<void>(), qnnTensor_);

  isAlloc_ = true;
}

void QNNTensorWrapper::setScaleOffsetQuantization(const std::vector<Qnn_ScaleOffset_t>& scaleOffsets, int32_t axis) {
  scaleOffsets_ = scaleOffsets;
  qnnTensor_.v2.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
  qnnTensor_.v2.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
  qnnTensor_.v2.quantizeParams.axisScaleOffsetEncoding = Qnn_AxisScaleOffset_t{
      .axis = axis,
      .numScaleOffsets = (uint32_t)scaleOffsets_.size(),
      .scaleOffset = scaleOffsets_.data(),
  };
}

void QNNTensorWrapper::setBlockwiseQuantization(const Qnn_BlockwiseExpansion_t& blockwise,
                                                const std::vector<Qnn_ScaleOffset_t>& scaleOffsets) {
  scaleOffsets_ = scaleOffsets;
  blockwiseExpansion_ = blockwise;

  blockwiseExpansion_.scaleOffsets = scaleOffsets_.data();

  qnnTensor_.v2.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
  qnnTensor_.v2.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION;
  qnnTensor_.v2.quantizeParams.blockwiseExpansion = &blockwiseExpansion_;
}

void QNNTensorWrapper::initFromQnnTensor(Qnn_Tensor_t* qnnTensor) {
  if (qnnTensor == nullptr) {
    MLLM_ERROR("QNNTensorWrapper::setQnnTensor() received nullptr");
    return;
  }

  // Update wrapper's internal state based on the provided tensor
  name_ = QNN_TENSOR_GET_NAME(qnnTensor) ? QNN_TENSOR_GET_NAME(qnnTensor) : "";

  // Update dimensions vector
  dimensions_.clear();
  uint32_t rank = QNN_TENSOR_GET_RANK(qnnTensor);
  dimensions_.reserve(rank);
  for (uint32_t i = 0; i < rank; ++i) { dimensions_.push_back(QNN_TENSOR_GET_DIMENSIONS(qnnTensor)[i]); }

  // Instead of deep copying, we'll do a shallow copy and manage the critical fields ourselves
  // This avoids unnecessary memory allocation and potential memory leaks
  qnnTensor_ = *qnnTensor;  // Shallow copy

  // Override the name and dimensions pointers to use our managed storage
  qnnTensor_.v2.name = name_.c_str();
  qnnTensor_.v2.dimensions = dimensions_.data();
}

std::shared_ptr<QNNParamTensorWrapper> QNNParamTensorWrapper::create(const std::string& paramName,
                                                                     const std::string& tensorName, Qnn_DataType_t dataType,
                                                                     const std::vector<uint32_t>& dimensions) {
  return std::make_shared<QNNParamTensorWrapper>(paramName, tensorName, dataType, dimensions);
}

QNNParamTensorWrapper::QNNParamTensorWrapper(const std::string& paramName, const std::string& tensorName,
                                             Qnn_DataType_t dataType, const std::vector<uint32_t>& dimensions) {
  paramName_ = paramName;
  tensorName_ = tensorName;
  dimensions_ = dimensions;
  // Fix parameters.
  qnnParam_.paramType = QNN_PARAMTYPE_TENSOR;
  qnnParam_.tensorParam.version = QNN_TENSOR_VERSION_2;
  qnnParam_.tensorParam.v2 = QNN_TENSOR_V2_INIT;
  qnnParam_.tensorParam.v2.type = QNN_TENSOR_TYPE_STATIC;
  qnnParam_.tensorParam.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  qnnParam_.tensorParam.v2.quantizeParams = DEFAULT_QUANTIZE_PARAMS;
  qnnParam_.tensorParam.v2.memType = QNN_TENSORMEMTYPE_RAW;
  // Custom parameters.
  qnnParam_.name = paramName_.c_str();
  qnnParam_.tensorParam.v2.name = tensorName_.c_str();
  qnnParam_.tensorParam.v2.dataType = dataType;
  qnnParam_.tensorParam.v2.rank = dimensions_.size();
  qnnParam_.tensorParam.v2.dimensions = dimensions_.data();
  qnnParam_.tensorParam.v2.clientBuf = {.data = nullptr, .dataSize = 0};
}

QNNParamTensorWrapper::~QNNParamTensorWrapper() {
  MLLM_RT_ASSERT(QNN_TENSOR_GET_CLIENT_BUF(qnnParam_.tensorParam).data != nullptr);
  free(QNN_TENSOR_GET_CLIENT_BUF(qnnParam_.tensorParam).data);
}
void* QNNParamTensorWrapper::alloc() {
  uint32_t dataSize = qnnDataTypeToSize(QNN_TENSOR_GET_DATA_TYPE(qnnParam_.tensorParam));
  for (int i = 0; i < QNN_TENSOR_GET_RANK(qnnParam_.tensorParam); i++) { dataSize *= qnnParam_.tensorParam.v2.dimensions[i]; }
  Qnn_ClientBuffer_t clientBuffer = {.data = malloc(dataSize), .dataSize = dataSize};
  QNN_TENSOR_SET_CLIENT_BUF(qnnParam_.tensorParam, clientBuffer);
  MLLM_RT_ASSERT(QNN_TENSOR_GET_CLIENT_BUF(qnnParam_.tensorParam).data != nullptr);
  return QNN_TENSOR_GET_CLIENT_BUF(qnnParam_.tensorParam).data;
}

Qnn_Param_t* QNNParamScalarWrapper::getNativeParam() { return &(qnnParam_); }

// --------------- QNN Graph Output Helper ---------------

Qnn_TensorType_t getQnnOutputTensorType(const std::shared_ptr<mllm::ir::tensor::TensorValue>& tensorValue) {
  if (tensorValue->getAttr("is_graph_output")) { return QNN_TENSOR_TYPE_APP_READ; }
  return QNN_TENSOR_TYPE_NATIVE;
}

// --------------- QNN Quantization Helper ---------------

Qnn_QuantizeParams_t createQuantizeParams(const Tensor& tensor) {
  auto quantParam = DEFAULT_QUANTIZE_PARAMS;
  if (tensor.dtype() == kInt8 || tensor.dtype() == kInt16) {
    quantParam = {QNN_DEFINITION_DEFINED,
                  QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                  {.scaleOffsetEncoding = {.scale = getQuantScale(const_cast<Tensor&>(tensor)), .offset = 0}}};
  }
  return quantParam;
}

void propagateQuantScale(const Tensor& input, Tensor& output) {
  if (input.dtype() == kInt8 || input.dtype() == kInt16) {
    // IMPORTANT! propagate quantization scale from input to output for afterward ops
    auto t = input;  // shadow copy for get scale
    setQuantScale(output, getQuantScale(t));
  }
}

void __printQnnTensor(const Qnn_Tensor_t* tensor) {
  if (tensor == nullptr) {
    MLLM_ERROR("Tensor is null");
    return;
  }
  if (tensor->version != QNN_TENSOR_VERSION_2) {
    MLLM_ERROR("Only Qnn_TensorV2_t is supported");
    return;
  }

  const Qnn_TensorV2_t& t = tensor->v2;

  std::string tensor_type = "";

  switch (t.type) {
    case QNN_TENSOR_TYPE_APP_READ: tensor_type = "APP_READ"; break;
    case QNN_TENSOR_TYPE_APP_WRITE: tensor_type = "APP_WRITE"; break;
    case QNN_TENSOR_TYPE_NATIVE: tensor_type = "APP_NATIVE"; break;
    case QNN_TENSOR_TYPE_STATIC: tensor_type = "STATIC"; break;
    default: tensor_type = "UNKNOWN";
  }

  std::string dtype_str;
  switch (t.dataType) {
    case QNN_DATATYPE_INT_8: dtype_str = "INT_8"; break;
    case QNN_DATATYPE_INT_16: dtype_str = "INT_16"; break;
    case QNN_DATATYPE_INT_32: dtype_str = "INT_32"; break;
    case QNN_DATATYPE_INT_64: dtype_str = "INT_64"; break;
    case QNN_DATATYPE_UINT_8: dtype_str = "UINT_8"; break;
    case QNN_DATATYPE_UINT_16: dtype_str = "UINT_16"; break;
    case QNN_DATATYPE_UINT_32: dtype_str = "UINT_32"; break;
    case QNN_DATATYPE_UINT_64: dtype_str = "UINT_64"; break;
    case QNN_DATATYPE_FLOAT_16: dtype_str = "FLOAT_16"; break;
    case QNN_DATATYPE_FLOAT_32: dtype_str = "FLOAT_32"; break;
    case QNN_DATATYPE_FLOAT_64: dtype_str = "FLOAT_64"; break;
    case QNN_DATATYPE_SFIXED_POINT_4: dtype_str = "SFIXED_POINT_4"; break;
    case QNN_DATATYPE_SFIXED_POINT_8: dtype_str = "SFIXED_POINT_8"; break;
    case QNN_DATATYPE_SFIXED_POINT_16: dtype_str = "SFIXED_POINT_16"; break;
    case QNN_DATATYPE_SFIXED_POINT_32: dtype_str = "SFIXED_POINT_32"; break;
    case QNN_DATATYPE_UFIXED_POINT_4: dtype_str = "UFIXED_POINT_4"; break;
    case QNN_DATATYPE_UFIXED_POINT_8: dtype_str = "UFIXED_POINT_8"; break;
    case QNN_DATATYPE_UFIXED_POINT_16: dtype_str = "UFIXED_POINT_16"; break;
    case QNN_DATATYPE_UFIXED_POINT_32: dtype_str = "UFIXED_POINT_32"; break;
    case QNN_DATATYPE_BOOL_8: dtype_str = "BOOL_8"; break;
    case QNN_DATATYPE_STRING: dtype_str = "STRING"; break;
    default: dtype_str = "UNKNOWN"; break;
  }

  std::string shape_str = "[";
  for (uint32_t i = 0; i < t.rank; ++i) {
    shape_str += std::to_string(t.dimensions[i]);
    if (i < t.rank - 1) shape_str += ", ";
  }
  shape_str += "]";

  std::string quant_str = "None";
  if (t.quantizeParams.encodingDefinition == QNN_DEFINITION_DEFINED) {
    if (t.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
      quant_str = "Scale: " + std::to_string(t.quantizeParams.scaleOffsetEncoding.scale)
                  + ", Offset: " + std::to_string(t.quantizeParams.scaleOffsetEncoding.offset);
    } else if (t.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
      quant_str = "Axis Scale Offset (Axis: " + std::to_string(t.quantizeParams.axisScaleOffsetEncoding.axis) + ")";
    } else if (t.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION) {
      quant_str = "Blockwise Expansion (axis:" + std::to_string(t.quantizeParams.blockwiseExpansion->axis)
                  + ", blockSize:" + std::to_string(t.quantizeParams.blockwiseExpansion->numBlocksPerAxis) + ")";
    }
  }

  MLLM_INFO("Tensor: {}, Type:{}, Shape: {}, Dtype: {}, Quant: {}", t.name, tensor_type, shape_str, dtype_str, quant_str);
}

}  // namespace mllm::qnn
