
#pragma once

#include "QnnTypes.h"

inline void setQnnTensorId(Qnn_Tensor_t& tensor, const uint32_t id) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.id = id;
}

inline void setQnnTensorId(Qnn_Tensor_t* tensor, uint32_t id) { setQnnTensorId(*tensor, id); }

inline void setQnnTensorName(Qnn_Tensor_t& tensor, const char* const name) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.name = name;
}

inline void setQnnTensorName(Qnn_Tensor_t* tensor, const char* name) { setQnnTensorName(*tensor, name); }

inline void setQnnTensorType(Qnn_Tensor_t& tensor, Qnn_TensorType_t type) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.type = type;
}

inline void setQnnTensorType(Qnn_Tensor_t* tensor, Qnn_TensorType_t type) { setQnnTensorType(*tensor, type); }

inline void setQnnTensorDataFormat(Qnn_Tensor_t& tensor, const Qnn_TensorDataFormat_t dataFormat) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.dataFormat = dataFormat;
}

inline void setQnnTensorDataFormat(Qnn_Tensor_t* tensor, Qnn_TensorDataFormat_t format) {
  setQnnTensorDataFormat(*tensor, format);
}

inline void setQnnTensorDataType(Qnn_Tensor_t& tensor, const Qnn_DataType_t dataType) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.dataType = dataType;
}

inline void setQnnTensorDataType(Qnn_Tensor_t* tensor, Qnn_DataType_t dataType) { setQnnTensorDataType(*tensor, dataType); }

inline void setQnnTensorQuantParams(Qnn_Tensor_t& tensor, const Qnn_QuantizeParams_t quantizeParams) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.quantizeParams = quantizeParams;
}

inline void setQnnTensorQuantParams(Qnn_Tensor_t* tensor, Qnn_QuantizeParams_t params) {
  setQnnTensorQuantParams(*tensor, params);
}

inline void setQnnTensorRank(Qnn_Tensor_t& tensor, const uint32_t rank) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.rank = rank;
}

inline void setQnnTensorRank(Qnn_Tensor_t* tensor, uint32_t rank) { setQnnTensorRank(*tensor, rank); }

inline void setQnnTensorDimensions(Qnn_Tensor_t& tensor, uint32_t* const dimensions) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.dimensions = dimensions;
}

inline void setQnnTensorDimensions(Qnn_Tensor_t* tensor, uint32_t* dims) { setQnnTensorDimensions(*tensor, dims); }

inline void setQnnTensorIsDynamicDimensions(Qnn_Tensor_t& tensor, uint8_t* isDynamic) {
  if (tensor.version == QNN_TENSOR_VERSION_2) { tensor.v2.isDynamicDimensions = isDynamic; }
}

inline void setQnnTensorIsDynamicDimensions(Qnn_Tensor_t* tensor, uint8_t* isDynamic) {
  setQnnTensorIsDynamicDimensions(*tensor, isDynamic);
}

inline void setQnnTensorSparseParams(Qnn_Tensor_t& tensor, Qnn_SparseParams_t sparseParams) {
  if (tensor.version == QNN_TENSOR_VERSION_2) { tensor.v2.sparseParams = sparseParams; }
}

inline void setQnnTensorSparseParams(Qnn_Tensor_t* tensor, Qnn_SparseParams_t sparseParams) {
  setQnnTensorSparseParams(*tensor, sparseParams);
}

inline void setQnnTensorMemType(Qnn_Tensor_t& tensor, const Qnn_TensorMemType_t memType) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.memType = memType;
}

inline void setQnnTensorMemType(Qnn_Tensor_t* tensor, Qnn_TensorMemType_t memType) { setQnnTensorMemType(*tensor, memType); }

inline void setQnnTensorClientBuf(Qnn_Tensor_t& tensor, const Qnn_ClientBuffer_t clientBuf) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.clientBuf = clientBuf;
}

inline void setQnnTensorClientBuf(Qnn_Tensor_t* tensor, Qnn_ClientBuffer_t clientBuf) {
  setQnnTensorClientBuf(*tensor, clientBuf);
}

inline void setQnnTensorMemHandle(Qnn_Tensor_t& tensor, const Qnn_MemHandle_t memHandle) {
  // TensorCompatTest justifies no need to check version
  tensor.v1.memHandle = memHandle;
}

inline void setQnnTensorMemHandle(Qnn_Tensor_t* tensor, Qnn_MemHandle_t handle) { setQnnTensorMemHandle(*tensor, handle); }

inline void setQnnTensorClientBufRetrieve(Qnn_Tensor_t& tensor, Qnn_TensorRetrieveRaw_t* const retrieve) {
  if (tensor.version == QNN_TENSOR_VERSION_2) { tensor.v2.retrieveRaw = retrieve; }
}
inline void setQnnTensorClientBufRetrieve(Qnn_Tensor_t* const tensor, Qnn_TensorRetrieveRaw_t* const retrieve) {
  setQnnTensorClientBufRetrieve(*tensor, retrieve);
}
inline void setQnnTensorClientBufRetrieve(Qnn_Tensor_t& tensor, Qnn_TensorRetrieveRaw_t& retrieve) {
  setQnnTensorClientBufRetrieve(tensor, &retrieve);
}
inline void setQnnTensorClientBufRetrieve(Qnn_Tensor_t* const tensor, Qnn_TensorRetrieveRaw_t& retrieve) {
  setQnnTensorClientBufRetrieve(*tensor, &retrieve);
}

// Accessors for QNN Op Config
// #define QNN_OP_CFG_GET_NAME(opConfig) getQnnOpConfigName(opConfig)
#define QNN_OP_CFG_GET_PACKAGE_NAME(opConfig) getQnnOpConfigPackageName(opConfig)
#define QNN_OP_CFG_GET_TYPE_NAME(opConfig) getQnnOpConfigTypeName(opConfig)
#define QNN_OP_CFG_GET_NUM_PARAMS(opConfig) getQnnOpConfigNumParams(opConfig)
#define QNN_OP_CFG_GET_PARAMS(opConfig) getQnnOpConfigParams(opConfig)
#define QNN_OP_CFG_GET_NUM_INPUTS(opConfig) getQnnOpConfigNumInputs(opConfig)
#define QNN_OP_CFG_GET_INPUTS(opConfig) getQnnOpConfigInputs(opConfig)
#define QNN_OP_CFG_GET_NUM_OUTPUTS(opConfig) getQnnOpConfigNumOutputs(opConfig)
#define QNN_OP_CFG_GET_OUTPUTS(opConfig) getQnnOpConfigOutputs(opConfig)

// Modifiers for QNN Op Config
#define QNN_OP_CFG_SET_NAME(opConfig, value) setQnnOpConfigName(opConfig, value)
#define QNN_OP_CFG_SET_PACKAGE_NAME(opConfig, value) setQnnOpConfigPackageName(opConfig, value)
#define QNN_OP_CFG_SET_TYPE_NAME(opConfig, value) setQnnOpConfigTypeName(opConfig, value)
#define QNN_OP_CFG_SET_PARAMS(opConfig, numOfParams, params) setQnnOpConfigParams(opConfig, numOfParams, params)
#define QNN_OP_CFG_SET_INPUTS(opConfig, numOfInputs, inputTensors) setQnnOpConfigInputs(opConfig, numOfInputs, inputTensors)
#define QNN_OP_CFG_SET_OUTPUTS(opConfig, numOfOutputs, outputTensors) \
  setQnnOpConfigOutputs(opConfig, numOfOutputs, outputTensors)

// Creator for QNN Tensor
#define QNN_TENSOR_CREATE(version) createQnnTensor(version)

// Accessors for QNN Tensor
#define QNN_TENSOR_GET_ID(tensor) getQnnTensorId(tensor)
#define QNN_TENSOR_GET_NAME(tensor) getQnnTensorName(tensor)
#define QNN_TENSOR_GET_TYPE(tensor) getQnnTensorType(tensor)
#define QNN_TENSOR_GET_DATA_FORMAT(tensor) getQnnTensorDataFormat(tensor)
#define QNN_TENSOR_GET_DATA_TYPE(tensor) getQnnTensorDataType(tensor)
#define QNN_TENSOR_GET_QUANT_PARAMS(tensor) getQnnTensorQuantParams(tensor)
#define QNN_TENSOR_GET_RANK(tensor) getQnnTensorRank(tensor)
#define QNN_TENSOR_GET_DIMENSIONS(tensor) getQnnTensorDimensions(tensor)
#define QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(tensor) getQnnTensorIsDynamicDimensions(tensor)
#define QNN_TENSOR_GET_SPARSE_PARAMS(tensor) getQnnTensorSparseParams(tensor)
#define QNN_TENSOR_GET_MEM_TYPE(tensor) getQnnTensorMemType(tensor)
#define QNN_TENSOR_GET_CLIENT_BUF(tensor) getQnnTensorClientBuf(tensor)
#define QNN_TENSOR_GET_MEM_HANDLE(tensor) getQnnTensorMemHandle(tensor)
#define QNN_TENSOR_GET_CLIENT_BUF_RETRIEVE(tensor) getQnnTensorClientBufRetrieve(tensor)

// Modifiers for QNN Tensor
#define QNN_TENSOR_SET_ID(tensor, value) setQnnTensorId(tensor, value)
#define QNN_TENSOR_SET_NAME(tensor, value) setQnnTensorName(tensor, value)
#define QNN_TENSOR_SET_TYPE(tensor, value) setQnnTensorType(tensor, value)
#define QNN_TENSOR_SET_DATA_FORMAT(tensor, value) setQnnTensorDataFormat(tensor, value)
#define QNN_TENSOR_SET_DATA_TYPE(tensor, value) setQnnTensorDataType(tensor, value)
#define QNN_TENSOR_SET_QUANT_PARAMS(tensor, value) setQnnTensorQuantParams(tensor, value)
#define QNN_TENSOR_SET_RANK(tensor, value) setQnnTensorRank(tensor, value)
#define QNN_TENSOR_SET_DIMENSIONS(tensor, value) setQnnTensorDimensions(tensor, value)
#define QNN_TENSOR_SET_IS_DYNAMIC_DIMENSIONS(tensor, value) setQnnTensorIsDynamicDimensions(tensor, value)
#define QNN_TENSOR_SET_SPARSE_PARAMS(tensor, value) setQnnTensorSparseParams(tensor, value)
#define QNN_TENSOR_SET_MEM_TYPE(tensor, value) setQnnTensorMemType(tensor, value)
#define QNN_TENSOR_SET_CLIENT_BUF(tensor, value) setQnnTensorClientBuf(tensor, value)
#define QNN_TENSOR_SET_MEM_HANDLE(tensor, value) setQnnTensorMemHandle(tensor, value)
#define QNN_TENSOR_SET_CLIENT_BUF_RETRIEVE(tensor, value) setQnnTensorClientBufRetrieve(tensor, value)
