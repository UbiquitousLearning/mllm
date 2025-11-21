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
#include <cstdint>
#include <memory>
#include <dlfcn.h>
#include <cstring>

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
  if (type != QNN_TENSOR_TYPE_STATIC) { MLLM_RT_ASSERT(tensor.device() == kQNN); }

  Qnn_DataType_t dataType = QNN_DATATYPE_UNDEFINED;
  switch (tensor.dtype()) {
    case kFloat32: dataType = QNN_DATATYPE_FLOAT_32; break;
    case kFloat16: dataType = QNN_DATATYPE_FLOAT_16; break;
    case kInt8: dataType = QNN_DATATYPE_SFIXED_POINT_8; break;
    case kInt16: dataType = QNN_DATATYPE_SFIXED_POINT_16; break;
    case kInt32: dataType = QNN_DATATYPE_SFIXED_POINT_32; break;
    case kUInt8: dataType = QNN_DATATYPE_UFIXED_POINT_8; break;
    default: MLLM_ERROR("Unsupported tensor element type for QNN: {}", (int)tensor.dtype()); break;
  }

  std::vector<uint32_t> dimensions(tensor.rank());
  for (int i = 0; i < tensor.rank(); i++) { dimensions[i] = tensor.shape()[i]; }

  auto tensorWrapper = std::make_shared<QNNTensorWrapper>(name, type, dataType, dimensions, quantize);

  tensorWrapper->dataContainer_ = tensor;

  return tensorWrapper;
}

std::shared_ptr<QNNTensorWrapper> QNNTensorWrapper::createStaticTensor(const std::string& name, const Tensor& tensor,
                                                                       Qnn_QuantizeParams_t quantize) {
  MLLM_RT_ASSERT(!name.empty() && tensor.rank() > 0 && !tensor.isNil());

  // mllm currently support float16/float32/sfixed8(int8) as static tensor (weight) data type
  // uint8 and int32 is caused by QNNLinear which uses Conv2d
  MLLM_RT_ASSERT(tensor.dtype() == kFloat16 || tensor.dtype() == kFloat32 || tensor.dtype() == kInt8 || tensor.dtype() == kUInt8
                 || tensor.dtype() == kInt32);

  std::shared_ptr<QNNTensorWrapper> tensorWrapper = QNNTensorWrapper::create(name, QNN_TENSOR_TYPE_STATIC, tensor, quantize);

  tensorWrapper->isAlloc_ = true;
  tensorWrapper->registeredPtr_ = tensor.ptr<void>();

  uint32_t numElement = tensor.bytes();
  Qnn_ClientBuffer_t clientBuffer = {.data = tensor.ptr<void>(), .dataSize = numElement};

  QNN_TENSOR_SET_CLIENT_BUF(tensorWrapper->qnnTensor_, clientBuffer);
  return tensorWrapper;
}

void QNNTensorWrapper::alloc() {
  MLLM_RT_ASSERT(dataContainer_.device() == kQNN);

  void* currentPtr = dataContainer_.impl()->ptr<void>();
  if (!currentPtr) {
    dataContainer_.alloc();
    currentPtr = dataContainer_.ptr<void>();
  }

  auto allocator = std::static_pointer_cast<QNNAllocator>(Context::instance().getBackend(kQNN)->allocator());

  auto storage = dataContainer_.impl()->storage();
  MLLM_RT_ASSERT(storage != nullptr);

  size_t requiredBytes = dataContainer_.bytes();

  // Check if we have a previously registered buffer pointer
  // This handles the case where tensor dimensions change (e.g., in decode phase)
  // and the existing registered buffer is too small
  if (registeredPtr_) {
    // Verify that the registered buffer is still valid
    if (!allocator->isRegistered(registeredPtr_)) {
      // Buffer was de-registered, clear the reference
      registeredPtr_ = nullptr;
      isAlloc_ = false;
    } else {
      // Check if the registered buffer is large enough for current requirements
      // If not, we need to de-register it and allocate a new one
      size_t registeredBytes = allocator->getRegisteredBufferSize(registeredPtr_);
      if (registeredBytes > 0 && registeredBytes < requiredBytes) {
        // Registered buffer is too small, de-register it
        // A new buffer will be allocated and registered below
        allocator->deRegisterQnnTensorFromSharedBuffer(registeredPtr_);
        registeredPtr_ = nullptr;
        isAlloc_ = false;
      }
    }
  }

  if (registeredPtr_ && registeredPtr_ != storage->ptr_) {
    if (!allocator->isRegistered(registeredPtr_)) {
      registeredPtr_ = nullptr;
    } else {
      void* freshPtr = storage->ptr_;
      size_t bytesToCopy = dataContainer_.bytes();
      if (freshPtr && bytesToCopy > 0) { std::memcpy(registeredPtr_, freshPtr, bytesToCopy); }
      if (freshPtr) { allocator->free(storage.get()); }
      storage->ptr_ = registeredPtr_;
      currentPtr = registeredPtr_;
    }
  }

  if (isAlloc_ && registeredPtr_ == currentPtr) { return; }

  if (!allocator->registerQnnTensorToSharedBuffer(storage.get(), qnnTensor_)) {
    MLLM_ERROR("QNNTensorWrapper::alloc failed to register shared buffer for tensor {}", name_);
    // Fail fast: prevent executing graph with invalid mem handle
    MLLM_RT_ASSERT(false);
  }

  registeredPtr_ = storage->ptr_;
  isAlloc_ = true;
}

void QNNTensorWrapper::resetAlloc() {
  isAlloc_ = false;
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
  uint32_t dataSize = QNNDataTypeToSize.find(QNN_TENSOR_GET_DATA_TYPE(qnnParam_.tensorParam))->second;
  for (int i = 0; i < QNN_TENSOR_GET_RANK(qnnParam_.tensorParam); i++) { dataSize *= qnnParam_.tensorParam.v2.dimensions[i]; }
  Qnn_ClientBuffer_t clientBuffer = {.data = malloc(dataSize), .dataSize = dataSize};
  QNN_TENSOR_SET_CLIENT_BUF(qnnParam_.tensorParam, clientBuffer);
  MLLM_RT_ASSERT(QNN_TENSOR_GET_CLIENT_BUF(qnnParam_.tensorParam).data != nullptr);
  return QNN_TENSOR_GET_CLIENT_BUF(qnnParam_.tensorParam).data;
}

QNNParamScalarWrapper::QNNParamScalarWrapper(const std::string& name, bool value) {
  name_ = name;
  qnnParam_.paramType = QNN_PARAMTYPE_SCALAR;
  qnnParam_.name = name_.c_str();
  qnnParam_.scalarParam.dataType = QNN_DATATYPE_BOOL_8;
  qnnParam_.scalarParam.bool8Value = static_cast<uint8_t>(value);
}

QNNParamScalarWrapper::QNNParamScalarWrapper(const std::string& name, uint32_t value) {
  name_ = name;
  qnnParam_.paramType = QNN_PARAMTYPE_SCALAR;
  qnnParam_.name = name_.c_str();
  qnnParam_.scalarParam.dataType = QNN_DATATYPE_UINT_32;
  qnnParam_.scalarParam.uint32Value = value;
}

QNNParamScalarWrapper::QNNParamScalarWrapper(const std::string& name, float value) {
  name_ = name;
  qnnParam_.paramType = QNN_PARAMTYPE_SCALAR;
  qnnParam_.name = name_.c_str();
  qnnParam_.scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
  qnnParam_.scalarParam.floatValue = value;
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

}  // namespace mllm::qnn
