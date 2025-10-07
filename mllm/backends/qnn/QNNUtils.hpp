#pragma once

#include "QnnCommon.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "QnnTypes.h"
#include "mllm/core/Tensor.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

// Forward declarations
namespace mllm::ir::tensor {
class TensorValue;
}

/**
 * @brief Utility functions for working with QNN tensors and QNN graphInfo structures.
 * @note It will NOT perform QNN checks, such as tensor version checks, etc.
 *       Currently, QNN tensor v1 and v2 are compatible for common variables.
 *       Future modifications should refer to $QNN_SDK_ROOT/examples/QNN/SampleApp
 */

namespace mllm::qnn {

#define CALL_QNN(apiCall)                                                                    \
  do {                                                                                       \
    int errorCode = ((apiCall) & 0xFFFF);                                                    \
    if (errorCode != QNN_SUCCESS) {                                                          \
      MLLM_ERROR("Error in file {}, line {}, error code {}", __FILE__, __LINE__, errorCode); \
      assert(errorCode == QNN_SUCCESS);                                                      \
    }                                                                                        \
  } while (0)

// --------------- Begin of QNN symbols loading ---------------

// func def for loading QNN Interface
using QnnInterfaceGetProvidersFn_t = Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*);
// func def of loading QNN System Interface
using QnnSystemInterfaceGetProvidersFn_t = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*);

extern QnnInterfaceGetProvidersFn_t QnnInterface_getProviders;
extern QnnSystemInterfaceGetProvidersFn_t QnnSystemInterface_getProviders;

bool loadQNNSymbol();
bool loadQNNSystemSymbol();

// --------------- End of QNN symbols loading ---------------

#define DEFAULT_QUANTIZE_PARAMS \
  (Qnn_QuantizeParams_t{        \
      QNN_DEFINITION_UNDEFINED, QNN_QUANTIZATION_ENCODING_UNDEFINED, {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}})

const std::map<Qnn_DataType_t, size_t> QNNDataTypeToSize = {
    {QNN_DATATYPE_INT_8, 1},           {QNN_DATATYPE_INT_16, 2},          {QNN_DATATYPE_INT_32, 4},
    {QNN_DATATYPE_INT_64, 8},          {QNN_DATATYPE_UINT_8, 1},          {QNN_DATATYPE_UINT_16, 2},
    {QNN_DATATYPE_UINT_32, 4},         {QNN_DATATYPE_UINT_64, 8},         {QNN_DATATYPE_FLOAT_16, 2},
    {QNN_DATATYPE_FLOAT_32, 4},        {QNN_DATATYPE_BOOL_8, 1},          {QNN_DATATYPE_SFIXED_POINT_8, 1},
    {QNN_DATATYPE_SFIXED_POINT_16, 2}, {QNN_DATATYPE_SFIXED_POINT_32, 4}, {QNN_DATATYPE_UFIXED_POINT_8, 1},
    {QNN_DATATYPE_UFIXED_POINT_16, 2}, {QNN_DATATYPE_UFIXED_POINT_32, 4},
};

// Utils for copying metadata to GraphInfo
using GraphInfo_t = struct GraphInfo {
  Qnn_GraphHandle_t graph;
  char* graphName;
  Qnn_Tensor_t* inputTensors;
  uint32_t numInputTensors;
  Qnn_Tensor_t* outputTensors;
  uint32_t numOutputTensors;
};
using GraphInfoPtr_t = GraphInfo_t*;

using GraphConfigInfo_t = struct GraphConfigInfo {
  char* graphName;
  const QnnGraph_Config_t** graphConfigs;
};

bool copyMetadataToGraphsInfo(const QnnSystemContext_BinaryInfo_t* binaryInfo, GraphInfo_t**& graphsInfo,
                              uint32_t& graphsCount);

bool copyGraphsInfo(const QnnSystemContext_GraphInfo_t* graphsInput, const uint32_t numGraphs, GraphInfo_t**& graphsInfo);

bool copyGraphsInfoV1(const QnnSystemContext_GraphInfoV1_t* graphInfoSrc, GraphInfo_t* graphInfoDst);

bool copyGraphsInfoV3(const QnnSystemContext_GraphInfoV3_t* graphInfoSrc, GraphInfo_t* graphInfoDst);

bool copyTensorsInfo(const Qnn_Tensor_t* tensorsInfoSrc, Qnn_Tensor_t*& tensorWrappers, uint32_t tensorsCount);

bool deepCopyQnnTensorInfo(Qnn_Tensor_t* dst, const Qnn_Tensor_t* src);

bool freeQnnTensor(Qnn_Tensor_t& tensor);

bool freeQnnTensors(Qnn_Tensor_t*& tensors, uint32_t numTensors);

inline void __mllmQnnLoggerCallback(const char* fmt, QnnLog_Level_t level, uint64_t times_tamp, va_list argp) {
  const char* level_str = "";
  const char* color_start = "";
  const char* color_end = "\033[0m";  // Reset color

  switch (level) {
    case QNN_LOG_LEVEL_ERROR:
      level_str = "[ERROR]";
      color_start = "\033[91m";  // Light red
      break;
    case QNN_LOG_LEVEL_WARN:
      level_str = "[WARN]";
      color_start = "\033[93m";  // Light yellow
      break;
    case QNN_LOG_LEVEL_INFO:
      level_str = "[INFO]";
      color_start = "\033[96m";  // Light cyan
      break;
    case QNN_LOG_LEVEL_DEBUG:
      level_str = "[DEBUG]";
      color_start = "\033[95m";  // Light magenta
      break;
    case QNN_LOG_LEVEL_VERBOSE:
      level_str = "[VERBOSE]";
      color_start = "\033[94m";  // Light blue
      break;
    case QNN_LOG_LEVEL_MAX:
      level_str = "[UNKNOWN]";
      color_start = "\033[37m";  // Light gray
      break;
  }

  double ms = (double)times_tamp / 1000000.0;

  {
    fprintf(stdout, "%s%s%s (%.1fms, %ld) ", color_start, level_str, color_end, ms, times_tamp);
    vfprintf(stdout, fmt, argp);
  }
}

// --------------- get/set quant scale for mllm::Tensor ---------------
// currently only consider per-tensor quantization
inline float getQuantScale(Tensor& tensor) {
  if (!tensor.attachedViews().contains("quant_scale")) { return 0.0f; }
  return tensor.attachedViews()["quant_scale"]->ptr<float>()[0];
}

inline void setQuantScale(Tensor& tensor, float scale) {
  if (!tensor.attachedViews().contains("quant_scale")) {
    auto t = Tensor::empty({1}, kFloat32, kCPU).alloc();
    t.at<float>({0}) = scale;
    tensor.attach("quant_scale", t.impl());
  } else {
    tensor.attachedViews()["quant_scale"]->ptr<float>()[0] = scale;
  }
}

// --------------- QNN Graph Output Helper ---------------
/**
 * @brief Determines the appropriate QNN tensor type based on graph output attribute
 * @param tensorValue The tensor value to check for graph output attribute
 * @return QNN_TENSOR_TYPE_APP_WRITE if marked as graph output, QNN_TENSOR_TYPE_NATIVE otherwise
 */
Qnn_TensorType_t getQnnOutputTensorType(const std::shared_ptr<mllm::ir::tensor::TensorValue>& tensorValue);

// --------------- QNN Wrapper ---------------
// QNN tensors' resource management is in C style. Wrap it in a C++ class
// related issue: dimension vector can only be destroyed after QNN graph is finalized
// reference from MNN (https://github.com/alibaba/MNN)

class QNNTensorWrapper {
 public:
  QNNTensorWrapper(const std::string& name, Qnn_TensorType_t type, Qnn_DataType_t dataType,
                   const std::vector<uint32_t>& dimensions, Qnn_QuantizeParams_t quantize);
  ~QNNTensorWrapper() = default;

  // create a QNN empty tensor from mllm::Tensor (used for Op input/output tensor)
  static std::shared_ptr<QNNTensorWrapper> create(const std::string& name, Qnn_TensorType_t type, const Tensor& tensor,
                                                  Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);

  // create a static QNN tensor from mllm::Tensor
  static std::shared_ptr<QNNTensorWrapper> createStaticTensor(const std::string& name, const Tensor& tensor,
                                                              Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);

  // shadow copy from existing QNN tensor
  // the src qnnTensor can be released after this
  void initFromQnnTensor(Qnn_Tensor_t* qnnTensor);

  Qnn_Tensor_t* getNativeTensor() { return &qnnTensor_; }
  [[nodiscard]] const Qnn_Tensor_t* getNativeTensor() const { return &qnnTensor_; }

  // alloc graph input/output tensor memory in QNN shared buffer
  void alloc();
  Tensor& getDataContainer() { return dataContainer_; }
  const std::vector<uint32_t>* getDimension() { return &dimensions_; }

 private:
  std::string name_;
  std::vector<uint32_t> dimensions_;
  Tensor dataContainer_;
  Qnn_Tensor_t qnnTensor_;
  bool isAlloc_ = false;
};

class QNNParamTensorWrapper {
 public:
  static std::shared_ptr<QNNParamTensorWrapper> create(const std::string& paramName, const std::string& tensorName,
                                                       Qnn_DataType_t dataType, const std::vector<int32_t>& dimensions) {
    std::vector<uint32_t> vec(dimensions.size());
    for (int i = 0; i < dimensions.size(); i++) { vec[i] = (uint32_t)dimensions[i]; }
    return std::make_shared<QNNParamTensorWrapper>(paramName, tensorName, dataType, vec);
  }

  static std::shared_ptr<QNNParamTensorWrapper> create(const std::string& paramName, const std::string& tensorName,
                                                       Qnn_DataType_t dataType, const std::vector<uint32_t>& dimensions);

  QNNParamTensorWrapper(const std::string& paramName, const std::string& tensorName, Qnn_DataType_t dataType,
                        const std::vector<uint32_t>& dimensions);

  ~QNNParamTensorWrapper();

  void* alloc();
  Qnn_Param_t* getNativeParam() { return &qnnParam_; }
  Qnn_Tensor_t* getNativeTensor() { return &qnnParam_.tensorParam; }

 private:
  std::string paramName_;
  std::string tensorName_;
  std::vector<uint32_t> dimensions_;
  Qnn_Param_t qnnParam_{};
};

class QNNParamScalarWrapper {
 public:
  template<typename T>
  static std::shared_ptr<QNNParamScalarWrapper> create(const std::string& name, T value) {
    return std::make_shared<QNNParamScalarWrapper>(name, value);
  };
  QNNParamScalarWrapper(const std::string& name, bool value);
  QNNParamScalarWrapper(const std::string& name, uint32_t value);
  QNNParamScalarWrapper(const std::string& name, float value);
  Qnn_Param_t* getNativeParam();

 private:
  std::string name_;
  Qnn_Param_t qnnParam_{};
};

}  // namespace mllm::qnn