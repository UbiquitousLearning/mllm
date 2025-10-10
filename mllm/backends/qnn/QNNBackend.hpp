#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "HTP/QnnHtpDevice.h"
#include "System/QnnSystemInterface.h"
#include "mllm/backends/base/Backend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/backends/qnn/QNNModel.hpp"
#include "mllm/mllm.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn {

static const std::string QNN_Custom_Op_Package = "LLaMAPackage";
static const std::string QNN_Context_File = "qnn_context.bin";

enum class ProfilingLevel { OFF, BASIC, DETAILED, INVALID };
class QNNPerf {
 public:
  static std::unique_ptr<QNNPerf> create(const QNN_INTERFACE_VER_TYPE* qnnInterface) {
    return std::make_unique<QNNPerf>(qnnInterface);
  }
  explicit QNNPerf(const QNN_INTERFACE_VER_TYPE* qnnInterface);
  ~QNNPerf();
  void setRpcLatencyAndPolling();
  void setPowerConfigBurst();
  void setPowerConfigBalanced();

 private:
  const QNN_INTERFACE_VER_TYPE* mQnnInterface = nullptr;
  QnnHtpDevice_PerfInfrastructure_t mPerfInfra{};
  uint32_t mPowerConfigId;
  QnnHtpPerfInfrastructure_PowerConfig_t mPowerConfigBurst{};
  QnnHtpPerfInfrastructure_PowerConfig_t mPowerConfigBalanced{};
};

class QNNRuntime {
  friend class QNNBackend;

 public:
  ~QNNRuntime();

  static std::unique_ptr<QNNRuntime> create(ProfilingLevel profilingLevel = ProfilingLevel::OFF,
                                            QnnLog_Level_t qnnLogLevel = QNN_LOG_LEVEL_WARN) {
    return std::unique_ptr<QNNRuntime>(initRuntime(profilingLevel, qnnLogLevel));
  }

  bool createContext(Qnn_ContextHandle_t& context, QnnContext_Config_t** contextConfig = nullptr);
  bool retrieveContext(Qnn_ContextHandle_t& context, std::vector<std::shared_ptr<QNNModel>>& qnnModels,
                       QnnContext_Config_t** contextConfig = nullptr);

 private:
  QNN_INTERFACE_VER_TYPE qnnInterface;
  QNN_SYSTEM_INTERFACE_VER_TYPE qnnSystemInterface;

  Qnn_LogHandle_t logHandle = nullptr;
  Qnn_BackendHandle_t backendHandle = nullptr;
  Qnn_DeviceHandle_t deviceHandle = nullptr;
  Qnn_ProfileHandle_t profileHandle = nullptr;

  QNNRuntime(QNN_INTERFACE_VER_TYPE qnnInterface, QNN_SYSTEM_INTERFACE_VER_TYPE qnnSystemInterface,
             Qnn_LogHandle_t qnnLogHandle, Qnn_BackendHandle_t qnnBackendHandle, Qnn_DeviceHandle_t qnnDeviceHandle,
             Qnn_ProfileHandle_t qnnProfileHandle = nullptr)
      : qnnInterface(qnnInterface),
        qnnSystemInterface(qnnSystemInterface),
        logHandle(qnnLogHandle),
        backendHandle(qnnBackendHandle),
        deviceHandle(qnnDeviceHandle),
        profileHandle(qnnProfileHandle) {}

  std::string getBackendBuildId(QNN_INTERFACE_VER_TYPE& qnnInterface) {
    char* backendBuildId{nullptr};
    if (QNN_SUCCESS != qnnInterface.backendGetBuildId((const char**)&backendBuildId)) {
      MLLM_ERROR_EXIT(1, "Unable to get build Id from the backend.");
    }
    return (backendBuildId == nullptr ? std::string("") : std::string(backendBuildId));
  }

  static QNNRuntime* initRuntime(ProfilingLevel profilingLevel, QnnLog_Level_t qnnLogLevel);
};

class QNNBackend final : public Backend {
 public:
  QNNBackend();

  bool isWeightOnDevice() override { return false; }

  // QNN Graph build interfaces
  std::shared_ptr<QNNModel> createQnnGraph(const std::string& graphName);

  void graphAddNode(const std::string& graphName, const std::string& nodeName, const std::string& nodeType,
                    const std::vector<std::string>& inputTensorNames, const std::vector<std::string>& outputTensorNames,
                    const std::vector<std::shared_ptr<QNNParamTensorWrapper>>& tensorParams,
                    const std::vector<std::shared_ptr<QNNParamScalarWrapper>>& scalarParams,
                    const std::string& packageName = "qti.aisw");

  bool graphFinalize(const std::string& graphName);

  void graphExecute(const std::string& graphName, std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);

  // Tensor management interfaces
  bool addTensor(const std::string& graphName, const std::string& tensorName, Qnn_TensorType_t type, const Tensor& tensor,
                 Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);

  bool addStaticTensor(const std::string& graphName, const std::string& tensorName, const Tensor& tensor,
                       Qnn_QuantizeParams_t quantize = DEFAULT_QUANTIZE_PARAMS);

  // Get tensor wrapper by name from specific graph
  std::shared_ptr<QNNTensorWrapper> getTensorWrapper(const std::string& graphName, const std::string& tensorName);

  // Getters for runtime components
  [[nodiscard]] const QNN_INTERFACE_VER_TYPE& qnnInterface() const { return runtime_->qnnInterface; }
  [[nodiscard]] Qnn_BackendHandle_t backendHandle() const { return runtime_->backendHandle; }
  [[nodiscard]] Qnn_ContextHandle_t context() const { return context_; }

 private:
  bool debug_;
  ProfilingLevel profilingLevel_;
  Qnn_ContextHandle_t context_ = nullptr;
  std::unique_ptr<QNNRuntime> runtime_;
  std::unique_ptr<QNNPerf> perf_;

  // Graph management
  std::map<std::string, int> qnnModelIndexMap_;
  std::vector<std::shared_ptr<QNNModel>> qnnModels_;
  int currentQnnModelIndex_ = -1;

  // Helper methods
  void extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle);
};

}  // namespace mllm::qnn
