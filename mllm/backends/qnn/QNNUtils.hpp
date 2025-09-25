#pragma once

#include "QnnCommon.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "QnnTypes.h"
#include "mllm/core/Tensor.hpp"

#include <memory>
#include <string>
#include <vector>

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

class QNNTensorWrapper {
 public:
  static std::shared_ptr<QNNTensorWrapper> create(const std::string& name, Qnn_TensorType_t type, Qnn_DataType_t dataType,
                                                  const std::vector<uint32_t>& dimensions,
                                                  Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);
  static std::shared_ptr<QNNTensorWrapper> create(const std::string& name, Qnn_TensorType_t type, Qnn_DataType_t dataType,
                                                  const std::vector<int>& dimensions,
                                                  Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);
  static std::shared_ptr<QNNTensorWrapper> createStaticFloatTensor(const std::string& name, Qnn_DataType_t dataType,
                                                                   const std::vector<uint32_t>& dimensions, const float* buffer,
                                                                   Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);
  static std::shared_ptr<QNNTensorWrapper> createStaticFloatTensor(const std::string& name, Qnn_DataType_t dataType,
                                                                   const std::vector<int>& dimensions, const float* buffer,
                                                                   Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);
  QNNTensorWrapper(const std::string& name, Qnn_TensorType_t type, Qnn_DataType_t dataType,
                   const std::vector<uint32_t>& dimensions, Qnn_QuantizeParams_t quantize);
  ~QNNTensorWrapper();
  Qnn_Tensor_t* getNativeTensor();
  [[nodiscard]] const Qnn_Tensor_t* getNativeTensor() const;
  void* alloc();
  std::shared_ptr<Tensor> getDataContainer();
  const std::vector<uint32_t>* getDimension();

 private:
  std::string mName;
  std::vector<uint32_t> mDimensions;
  std::shared_ptr<Tensor> mDataContainer;
  Qnn_Tensor_t mQnnTensor;
  bool mIsAlloc = false;
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
  switch (level) {
    case QNN_LOG_LEVEL_ERROR: level_str = "[ERROR]"; break;
    case QNN_LOG_LEVEL_WARN: level_str = "[WARN]"; break;
    case QNN_LOG_LEVEL_INFO: level_str = "[INFO]"; break;
    case QNN_LOG_LEVEL_DEBUG: level_str = "[DEBUG]"; break;
    case QNN_LOG_LEVEL_VERBOSE: level_str = "[VERBOSE]"; break;
    case QNN_LOG_LEVEL_MAX: level_str = "[UNKNOWN]"; break;
  }

  double ms = (double)times_tamp / 1000000.0;

  {
    fprintf(stdout, "%s (%.1fms, %ld) ", level_str, ms, times_tamp);
    vfprintf(stdout, fmt, argp);
  }
}

}  // namespace mllm::qnn