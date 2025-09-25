#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "HTP/QnnHtpDevice.h"
#include "System/QnnSystemInterface.h"
#include "mllm/backends/base/Backend.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/mllm.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn {

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
  bool retrieveContext(Qnn_ContextHandle_t& context, std::vector<GraphInfo_t*>& graphsInfo,
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

 private:
  bool debug_, isFromCache_ = false;
  ProfilingLevel profilingLevel_;
  Qnn_ContextHandle_t context_ = nullptr;
  std::unique_ptr<QNNRuntime> runtime_;
  std::unique_ptr<QNNPerf> perf_;

  std::vector<GraphInfo_t*> graphsInfo_;
  std::map<std::string, int> qnnModelIndexMap_;
};

}  // namespace mllm::qnn
