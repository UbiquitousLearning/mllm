// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <dlfcn.h>
#include <cstdlib>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <atomic>
#include <functional>
#include <unordered_map>

#include <QNN/QnnCommon.h>
#include <QNN/QnnContext.h>
#include <QNN/QnnInterface.h>
#include <QNN/QnnSdkBuildId.h>
#include <QNN/HTP/QnnHtpDevice.h>
#include <QNN/System/QnnSystemInterface.h>

#include "mllm/backends/qnn/aot/QnnTargetMachine.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::qnn::aot {

void __mllmLoggerCallback4QnnLogger(const char* fmt, QnnLog_Level_t level, uint64_t times_tamp, va_list argp);

// Collection of symbols that we need to load from qnn dyn lib.
struct QnnFuncSymbols {
  using QnnInterfaceGetProvidersFuncType = Qnn_ErrorHandle_t(const QnnInterface_t*** providerList, uint32_t* numProviders);
  using QnnSystemInterfaceGetProvidersFuncType = Qnn_ErrorHandle_t(const QnnSystemInterface_t*** providerList,
                                                                   uint32_t* numProviders);

  QNN_INTERFACE_VER_TYPE qnn_interface_;
  QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface_;
};

struct QnnDeviceAndContext {
  std::string name_;
  Qnn_LogHandle_t log_ = nullptr;
  Qnn_BackendHandle_t bk_handle_ = nullptr;
  Qnn_DeviceHandle_t device_handle_ = nullptr;
  QnnBackend_Config_t** bk_cfg_ = nullptr;
  QnnContext_Config_t** qnn_context_config_ = nullptr;
  Qnn_ProfileHandle_t profile_bk_handle_ = nullptr;
  Qnn_ContextHandle_t qnn_ctx_handle_;
};

struct QnnDynLibDescriptor {
  std::string lib_name_;
  std::string lib_path_;
  void* handle_ = nullptr;

  template<typename FuncType>
  std::function<FuncType> func(const std::string& symbol_name) {
    if (handle_ == nullptr) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "QnnDynSymbolLoader: handle is nullptr."); }
    auto func_ptr = dlsym(handle_, symbol_name.c_str());
    MLLM_RT_ASSERT(func_ptr != nullptr);
    return (FuncType*)(func_ptr);
  };
};

class QnnDynSymbolLoader {
 public:
  enum DynFlag : int {  // NOLINT performance-enum-size
    kRTLD_NOW = RTLD_NOW,
    kRTLD_LOCAL = RTLD_LOCAL,
    kRTLD_GLOBAL = RTLD_GLOBAL,
  };

  static QnnDynSymbolLoader& instance() {
    static QnnDynSymbolLoader instance;
    return instance;
  }

  ~QnnDynSymbolLoader();

  QnnDynSymbolLoader() = default;

  QnnDynSymbolLoader(const QnnDynSymbolLoader&) = delete;

  QnnDynSymbolLoader& operator=(const QnnDynSymbolLoader&) = delete;

  bool loadQnnDynLib(const std::string& lib_name, int flag);

  bool loadQnnDynLibAtPath(const std::string& path, const std::string& lib_name, int flag);

  inline QnnDynLibDescriptor& operator()(const std::string& lib_name) { return libs_.at(lib_name); }

 private:
  std::unordered_map<std::string, QnnDynLibDescriptor> libs_;
  static const std::vector<std::string> possible_qnn_dyn_lib_paths_;
};

// Device and Dynamic Lib included
class QnnAOTEnv {
 public:
  using ptr_t = std::shared_ptr<QnnAOTEnv>;

  explicit QnnAOTEnv(QcomTargetMachine& target_machine);

  QnnAOTEnv(const std::string& lib_path, QcomTargetMachine& target_machine);

  std::shared_ptr<QnnDeviceAndContext> createContext(const std::string& name, bool weights_sharing = false);

  void saveContext(const std::string& name, const std::string& path);

  void destroyContext(const std::string& name);

  // This is for All PUs, such as CPU, GPU, NPU
  std::vector<QnnDevice_PlatformInfo_t*> createDevicePlatformInfo();

  // This function is for NPU only.
  std::vector<QnnDevice_CustomConfig_t> createDecideCustomConfigInfo();

  std::vector<QnnContext_CustomConfig_t> createContextCustomConfig(bool weights_sharing);

 private:
  void _setup(const std::string& path = "");

  QcomTargetMachine target_machine_;
  QnnFuncSymbols qnn_htp_func_symbols_;
  std::unordered_map<std::string, std::shared_ptr<QnnDeviceAndContext>> contexts_;

  // device config for all to use
  std::vector<QnnDevice_Config_t> target_machine_qnn_config_;
  std::vector<const QnnDevice_Config_t*> target_machine_qnn_config_ptrs_;

  // void* handle that should be freed when QnnAOTEnv end
  std::vector<void*> unreachable_handel_;
};

}  // namespace mllm::qnn::aot
