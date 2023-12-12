//==============================================================================
//
//  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include "QnnTypes.h"

#define QNN_OP_CFG_VALID(opConfig) ((opConfig).version == QNN_OPCONFIG_VERSION_1)

inline const char* getQnnOpConfigName(const Qnn_OpConfig_t& opConfig) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    return opConfig.v1.name;
  }
  return NULL;
}

inline const char* getQnnOpConfigName(const Qnn_OpConfig_t* opConfig) {
  return getQnnOpConfigName(*opConfig);
}

inline const char* getQnnOpConfigPackageName(const Qnn_OpConfig_t& opConfig) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    return opConfig.v1.packageName;
  }
  return NULL;
}

inline const char* getQnnOpConfigPackageName(const Qnn_OpConfig_t* opConfig) {
  return getQnnOpConfigPackageName(*opConfig);
}

inline const char* getQnnOpConfigTypeName(const Qnn_OpConfig_t& opConfig) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    return opConfig.v1.typeName;
  }
  return NULL;
}

inline const char* getQnnOpConfigTypeName(const Qnn_OpConfig_t* opConfig) {
  return getQnnOpConfigTypeName(*opConfig);
}

inline uint32_t getQnnOpConfigNumParams(const Qnn_OpConfig_t& opConfig) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    return opConfig.v1.numOfParams;
  }
  return 0u;
}

inline uint32_t getQnnOpConfigNumParams(const Qnn_OpConfig_t* opConfig) {
  return getQnnOpConfigNumParams(*opConfig);
}

inline const Qnn_Param_t* getQnnOpConfigParams(const Qnn_OpConfig_t& opConfig) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    return opConfig.v1.params;
  }
  return NULL;
}

inline const Qnn_Param_t* getQnnOpConfigParams(const Qnn_OpConfig_t* opConfig) {
  return getQnnOpConfigParams(*opConfig);
}

inline uint32_t getQnnOpConfigNumInputs(const Qnn_OpConfig_t& opConfig) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    return opConfig.v1.numOfInputs;
  }
  return 0u;
}

inline uint32_t getQnnOpConfigNumInputs(const Qnn_OpConfig_t* opConfig) {
  return getQnnOpConfigNumInputs(*opConfig);
}

inline const Qnn_Tensor_t* getQnnOpConfigInputs(const Qnn_OpConfig_t& opConfig) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    return opConfig.v1.inputTensors;
  }
  return NULL;
}

inline const Qnn_Tensor_t* getQnnOpConfigInputs(const Qnn_OpConfig_t* opConfig) {
  return getQnnOpConfigInputs(*opConfig);
}

inline uint32_t getQnnOpConfigNumOutputs(const Qnn_OpConfig_t& opConfig) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    return opConfig.v1.numOfOutputs;
  }
  return 0u;
}

inline uint32_t getQnnOpConfigNumOutputs(const Qnn_OpConfig_t* opConfig) {
  return getQnnOpConfigNumOutputs(*opConfig);
}

inline const Qnn_Tensor_t* getQnnOpConfigOutputs(const Qnn_OpConfig_t& opConfig) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    return opConfig.v1.outputTensors;
  }
  return NULL;
}

inline const Qnn_Tensor_t* getQnnOpConfigOutputs(const Qnn_OpConfig_t* opConfig) {
  return getQnnOpConfigOutputs(*opConfig);
}

inline void setQnnOpConfigName(Qnn_OpConfig_t& opConfig, const char* name) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    opConfig.v1.name = name;
  }
}

inline void setQnnOpConfigName(Qnn_OpConfig_t* opConfig, const char* name) {
  setQnnOpConfigName(*opConfig, name);
}

inline void setQnnOpConfigPackageName(Qnn_OpConfig_t& opConfig, const char* packageName) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    opConfig.v1.packageName = packageName;
  }
}

inline void setQnnOpConfigPackageName(Qnn_OpConfig_t* opConfig, const char* packageName) {
  setQnnOpConfigPackageName(*opConfig, packageName);
}

inline void setQnnOpConfigTypeName(Qnn_OpConfig_t& opConfig, const char* typeName) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    opConfig.v1.typeName = typeName;
  }
}

inline void setQnnOpConfigTypeName(Qnn_OpConfig_t* opConfig, const char* typeName) {
  setQnnOpConfigTypeName(*opConfig, typeName);
}

inline void setQnnOpConfigParams(Qnn_OpConfig_t& opConfig,
                                 uint32_t numOfParams,
                                 Qnn_Param_t* params) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    opConfig.v1.numOfParams = numOfParams;
    opConfig.v1.params      = params;
  }
}

inline void setQnnOpConfigParams(Qnn_OpConfig_t* opConfig,
                                 uint32_t numOfParams,
                                 Qnn_Param_t* params) {
  setQnnOpConfigParams(*opConfig, numOfParams, params);
}

inline void setQnnOpConfigInputs(Qnn_OpConfig_t& opConfig,
                                 uint32_t numOfInputs,
                                 Qnn_Tensor_t* inputTensors) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    opConfig.v1.numOfInputs  = numOfInputs;
    opConfig.v1.inputTensors = inputTensors;
  }
}

inline void setQnnOpConfigInputs(Qnn_OpConfig_t* opConfig,
                                 uint32_t numOfInputs,
                                 Qnn_Tensor_t* inputTensors) {
  setQnnOpConfigInputs(*opConfig, numOfInputs, inputTensors);
}

inline void setQnnOpConfigOutputs(Qnn_OpConfig_t& opConfig,
                                  uint32_t numOfOutputs,
                                  Qnn_Tensor_t* outputTensors) {
  if (opConfig.version == QNN_OPCONFIG_VERSION_1) {
    opConfig.v1.numOfOutputs  = numOfOutputs;
    opConfig.v1.outputTensors = outputTensors;
  }
}

inline void setQnnOpConfigOutputs(Qnn_OpConfig_t* opConfig,
                                  uint32_t numOfOutputs,
                                  Qnn_Tensor_t* outputTensors) {
  setQnnOpConfigOutputs(*opConfig, numOfOutputs, outputTensors);
}

inline uint32_t getQnnTensorId(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    return tensor.v1.id;
  }
  return 0u;
}

inline uint32_t getQnnTensorId(const Qnn_Tensor_t* tensor) { return getQnnTensorId(*tensor); }

inline const char* getQnnTensorName(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    return tensor.v1.name;
  }
  return 0u;
}

inline const char* getQnnTensorName(const Qnn_Tensor_t* tensor) {
  return getQnnTensorName(*tensor);
}

inline Qnn_TensorType_t getQnnTensorType(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    return tensor.v1.type;
  }
  return QNN_TENSOR_TYPE_UNDEFINED;
}

inline Qnn_TensorType_t getQnnTensorType(const Qnn_Tensor_t* tensor) {
  return getQnnTensorType(*tensor);
}

inline Qnn_TensorDataFormat_t getQnnTensorDataFormat(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    return tensor.v1.dataFormat;
  }
  return QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
}

inline Qnn_TensorDataFormat_t getQnnTensorDataFormat(const Qnn_Tensor_t* tensor) {
  return getQnnTensorDataFormat(*tensor);
}

inline Qnn_DataType_t getQnnTensorDataType(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    return tensor.v1.dataType;
  }
  return QNN_DATATYPE_UNDEFINED;
}

inline Qnn_DataType_t getQnnTensorDataType(const Qnn_Tensor_t* tensor) {
  return getQnnTensorDataType(*tensor);
}

inline Qnn_QuantizeParams_t getQnnTensorQuantParams(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    return tensor.v1.quantizeParams;
  }
  return QNN_QUANTIZE_PARAMS_INIT;
}

inline Qnn_QuantizeParams_t getQnnTensorQuantParams(const Qnn_Tensor_t* tensor) {
  return getQnnTensorQuantParams(*tensor);
}

inline uint32_t getQnnTensorRank(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    return tensor.v1.rank;
  }
  return 0u;
}

inline uint32_t getQnnTensorRank(const Qnn_Tensor_t* tensor) { return getQnnTensorRank(*tensor); }

inline uint32_t* getQnnTensorDimensions(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    return tensor.v1.dimensions;
  }
  return NULL;
}

inline uint32_t* getQnnTensorDimensions(const Qnn_Tensor_t* tensor) {
  return getQnnTensorDimensions(*tensor);
}

inline Qnn_TensorMemType_t getQnnTensorMemType(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    return tensor.v1.memType;
  }
  return QNN_TENSORMEMTYPE_UNDEFINED;
}

inline Qnn_TensorMemType_t getQnnTensorMemType(const Qnn_Tensor_t* tensor) {
  return getQnnTensorMemType(*tensor);
}

inline Qnn_ClientBuffer_t getQnnTensorClientBuf(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    return tensor.v1.clientBuf;
  }
  return QNN_CLIENT_BUFFER_INIT;
}

inline Qnn_ClientBuffer_t getQnnTensorClientBuf(const Qnn_Tensor_t* tensor) {
  return getQnnTensorClientBuf(*tensor);
}

inline Qnn_MemHandle_t getQnnTensorMemHandle(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    return tensor.v1.memHandle;
  }
  return NULL;
}

inline Qnn_MemHandle_t getQnnTensorMemHandle(const Qnn_Tensor_t* tensor) {
  return getQnnTensorMemHandle(*tensor);
}

inline void setQnnTensorId(Qnn_Tensor_t& tensor, uint32_t id) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    tensor.v1.id = id;
  }
}

inline void setQnnTensorId(Qnn_Tensor_t* tensor, uint32_t id) { setQnnTensorId(*tensor, id); }

inline void setQnnTensorName(Qnn_Tensor_t& tensor, const char* name) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    tensor.v1.name = name;
  }
}

inline void setQnnTensorName(Qnn_Tensor_t* tensor, const char* name) {
  setQnnTensorName(*tensor, name);
}

inline void setQnnTensorType(Qnn_Tensor_t& tensor, Qnn_TensorType_t type) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    tensor.v1.type = type;
  }
}

inline void setQnnTensorType(Qnn_Tensor_t* tensor, Qnn_TensorType_t type) {
  setQnnTensorType(*tensor, type);
}

inline void setQnnTensorDataFormat(Qnn_Tensor_t& tensor, Qnn_TensorDataFormat_t format) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    tensor.v1.dataFormat = format;
  }
}

inline void setQnnTensorDataFormat(Qnn_Tensor_t* tensor, Qnn_TensorDataFormat_t format) {
  setQnnTensorDataFormat(*tensor, format);
}

inline void setQnnTensorDataType(Qnn_Tensor_t& tensor, Qnn_DataType_t dataType) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    tensor.v1.dataType = dataType;
  }
}

inline void setQnnTensorDataType(Qnn_Tensor_t* tensor, Qnn_DataType_t dataType) {
  setQnnTensorDataType(*tensor, dataType);
}

inline void setQnnTensorQuantParams(Qnn_Tensor_t& tensor, Qnn_QuantizeParams_t params) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    tensor.v1.quantizeParams = params;
  }
}

inline void setQnnTensorQuantParams(Qnn_Tensor_t* tensor, Qnn_QuantizeParams_t params) {
  setQnnTensorQuantParams(*tensor, params);
}

inline void setQnnTensorRank(Qnn_Tensor_t& tensor, uint32_t rank) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    tensor.v1.rank = rank;
  }
}

inline void setQnnTensorRank(Qnn_Tensor_t* tensor, uint32_t rank) {
  setQnnTensorRank(*tensor, rank);
}

inline void setQnnTensorDimensions(Qnn_Tensor_t& tensor, uint32_t* dims) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    tensor.v1.dimensions = dims;
  }
}

inline void setQnnTensorDimensions(Qnn_Tensor_t* tensor, uint32_t* dims) {
  setQnnTensorDimensions(*tensor, dims);
}

inline void setQnnTensorMemType(Qnn_Tensor_t& tensor, Qnn_TensorMemType_t memType) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    tensor.v1.memType = memType;
  }
}

inline void setQnnTensorMemType(Qnn_Tensor_t* tensor, Qnn_TensorMemType_t memType) {
  setQnnTensorMemType(*tensor, memType);
}

inline void setQnnTensorClientBuf(Qnn_Tensor_t& tensor, Qnn_ClientBuffer_t clientBuf) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    tensor.v1.clientBuf = clientBuf;
  }
}

inline void setQnnTensorClientBuf(Qnn_Tensor_t* tensor, Qnn_ClientBuffer_t clientBuf) {
  setQnnTensorClientBuf(*tensor, clientBuf);
}

inline void setQnnTensorMemHandle(Qnn_Tensor_t& tensor, Qnn_MemHandle_t handle) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    tensor.v1.memHandle = handle;
  }
}

inline void setQnnTensorMemHandle(Qnn_Tensor_t* tensor, Qnn_MemHandle_t handle) {
  setQnnTensorMemHandle(*tensor, handle);
}

// Accessors for QNN Op Config
#define QNN_OP_CFG_GET_NAME(opConfig)         getQnnOpConfigName(opConfig)
#define QNN_OP_CFG_GET_PACKAGE_NAME(opConfig) getQnnOpConfigPackageName(opConfig)
#define QNN_OP_CFG_GET_TYPE_NAME(opConfig)    getQnnOpConfigTypeName(opConfig)
#define QNN_OP_CFG_GET_NUM_PARAMS(opConfig)   getQnnOpConfigNumParams(opConfig)
#define QNN_OP_CFG_GET_PARAMS(opConfig)       getQnnOpConfigParams(opConfig)
#define QNN_OP_CFG_GET_NUM_INPUTS(opConfig)   getQnnOpConfigNumInputs(opConfig)
#define QNN_OP_CFG_GET_INPUTS(opConfig)       getQnnOpConfigInputs(opConfig)
#define QNN_OP_CFG_GET_NUM_OUTPUTS(opConfig)  getQnnOpConfigNumOutputs(opConfig)
#define QNN_OP_CFG_GET_OUTPUTS(opConfig)      getQnnOpConfigOutputs(opConfig)

// Modifiers for QNN Op Config
#define QNN_OP_CFG_SET_NAME(opConfig, value)         setQnnOpConfigName(opConfig, value)
#define QNN_OP_CFG_SET_PACKAGE_NAME(opConfig, value) setQnnOpConfigPackageName(opConfig, value)
#define QNN_OP_CFG_SET_TYPE_NAME(opConfig, value)    setQnnOpConfigTypeName(opConfig, value)
#define QNN_OP_CFG_SET_PARAMS(opConfig, numOfParams, params) \
  setQnnOpConfigParams(opConfig, numOfParams, params)
#define QNN_OP_CFG_SET_INPUTS(opConfig, numOfInputs, inputTensors) \
  setQnnOpConfigInputs(opConfig, numOfInputs, inputTensors)
#define QNN_OP_CFG_SET_OUTPUTS(opConfig, numOfOutputs, outputTensors) \
  setQnnOpConfigOutputs(opConfig, numOfOutputs, outputTensors)

// Accessors for QNN Tensor
#define QNN_TENSOR_GET_ID(tensor)           getQnnTensorId(tensor)
#define QNN_TENSOR_GET_NAME(tensor)         getQnnTensorName(tensor)
#define QNN_TENSOR_GET_TYPE(tensor)         getQnnTensorType(tensor)
#define QNN_TENSOR_GET_DATA_FORMAT(tensor)  getQnnTensorDataFormat(tensor)
#define QNN_TENSOR_GET_DATA_TYPE(tensor)    getQnnTensorDataType(tensor)
#define QNN_TENSOR_GET_QUANT_PARAMS(tensor) getQnnTensorQuantParams(tensor)
#define QNN_TENSOR_GET_RANK(tensor)         getQnnTensorRank(tensor)
#define QNN_TENSOR_GET_DIMENSIONS(tensor)   getQnnTensorDimensions(tensor)
#define QNN_TENSOR_GET_MEM_TYPE(tensor)     getQnnTensorMemType(tensor)
#define QNN_TENSOR_GET_CLIENT_BUF(tensor)   getQnnTensorClientBuf(tensor)
#define QNN_TENSOR_GET_MEM_HANDLE(tensor)   getQnnTensorMemHandle(tensor)

// Modifiers for QNN Tensor
#define QNN_TENSOR_SET_ID(tensor, value)           setQnnTensorId(tensor, value)
#define QNN_TENSOR_SET_NAME(tensor, value)         setQnnTensorName(tensor, value)
#define QNN_TENSOR_SET_TYPE(tensor, value)         setQnnTensorType(tensor, value)
#define QNN_TENSOR_SET_DATA_FORMAT(tensor, value)  setQnnTensorDataFormat(tensor, value)
#define QNN_TENSOR_SET_DATA_TYPE(tensor, value)    setQnnTensorDataType(tensor, value)
#define QNN_TENSOR_SET_QUANT_PARAMS(tensor, value) setQnnTensorQuantParams(tensor, value)
#define QNN_TENSOR_SET_RANK(tensor, value)         setQnnTensorRank(tensor, value)
#define QNN_TENSOR_SET_DIMENSIONS(tensor, value)   setQnnTensorDimensions(tensor, value)
#define QNN_TENSOR_SET_MEM_TYPE(tensor, value)     setQnnTensorMemType(tensor, value)
#define QNN_TENSOR_SET_CLIENT_BUF(tensor, value)   setQnnTensorClientBuf(tensor, value)
#define QNN_TENSOR_SET_MEM_HANDLE(tensor, value)   setQnnTensorMemHandle(tensor, value)
