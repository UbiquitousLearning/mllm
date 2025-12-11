// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/QnnTargetMachine.hpp"

namespace mllm::qnn::aot {

void __mllmLoggerCallback4QnnLogger(const char* fmt, QnnLog_Level_t level, uint64_t times_tamp, va_list argp) {
  const char* level_str = "";
  switch (level) {
    case QNN_LOG_LEVEL_ERROR: level_str = "[ERROR]  "; break;
    case QNN_LOG_LEVEL_WARN: level_str = "[WARN]   "; break;
    case QNN_LOG_LEVEL_INFO: level_str = "[INFO]   "; break;
    case QNN_LOG_LEVEL_DEBUG: level_str = "[DEBUG]  "; break;
    case QNN_LOG_LEVEL_VERBOSE: level_str = "[VERBOSE]"; break;
    case QNN_LOG_LEVEL_MAX: level_str = "[UNKNOWN]"; break;
  }

  double ms = (double)times_tamp / 1000000.0;

  {
    fprintf(stdout, "QnnLogger(%8.1fms, %ld) %s: ", ms, times_tamp, level_str);
    vfprintf(stdout, fmt, argp);
  }
}

const std::vector<std::string> QnnDynSymbolLoader::possible_qnn_dyn_lib_paths_ = {
    "/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/",
};

QnnDynSymbolLoader::~QnnDynSymbolLoader() {
  for (auto& item : libs_) {
    if (item.second.handle_) { dlclose(item.second.handle_); }
  }
}

bool QnnDynSymbolLoader::loadQnnDynLib(const std::string& lib_name, int flag) {
  for (auto const& path : possible_qnn_dyn_lib_paths_) {
    auto real_path = path + lib_name;
    auto handle = dlopen(real_path.c_str(), flag);
    if (handle) {
      auto descriptor = QnnDynLibDescriptor{.lib_name_ = lib_name, .lib_path_ = path, .handle_ = handle};
      libs_.insert({lib_name, descriptor});
      MLLM_INFO("QnnDynSymbolLoader::loadQnnDynLib {} success.", real_path);
      return true;
    } else {
      char* error = dlerror();
      MLLM_ERROR("QnnDynSymbolLoader::loadQnnDynLib try for {} failed: {}", real_path, error ? error : "Unknown error");
    }
  }
  MLLM_ERROR("QnnDynSymbolLoader::loadQnnDynLib {} failed.", lib_name);
  return false;
}

bool QnnDynSymbolLoader::loadQnnDynLibAtPath(const std::string& path, const std::string& lib_name, int flag) {
  auto real_path = path + lib_name;
  auto handle = dlopen(real_path.c_str(), flag);
  if (handle) {
    auto descriptor = QnnDynLibDescriptor{.lib_name_ = lib_name, .lib_path_ = path, .handle_ = handle};
    libs_.insert({lib_name, descriptor});
    MLLM_INFO("QnnDynSymbolLoader::loadQnnDynLib {} success.", real_path);
    return true;
  } else {
    char* error = dlerror();
    MLLM_ERROR("QnnDynSymbolLoader::loadQnnDynLib try for {} failed: {}", real_path, error ? error : "Unknown error");
  }
  MLLM_ERROR("QnnDynSymbolLoader::loadQnnDynLib {} failed.", lib_name);
  return false;
}

QnnAOTEnv::QnnAOTEnv(QcomTargetMachine& target_machine) : target_machine_(target_machine) { _setup(); }

QnnAOTEnv::QnnAOTEnv(const std::string& lib_path, QcomTargetMachine& target_machine) : target_machine_(target_machine) {
  _setup(lib_path);
}

void QnnAOTEnv::_setup(const std::string& path) {
  auto& loader = QnnDynSymbolLoader::instance();
  std::string htp_backend_lib_name = "libQnnHtp.so";
  // GLOBAL Load
  if (path.empty()) {
    if (!loader.loadQnnDynLib(htp_backend_lib_name,
                              QnnDynSymbolLoader::DynFlag::kRTLD_NOW | QnnDynSymbolLoader::DynFlag::kRTLD_GLOBAL)) {
      MLLM_ERROR("QnnAOTEnv::QnnAOTEnv {} failed.", htp_backend_lib_name);
      exit(1);
    }
  } else {
    if (!loader.loadQnnDynLibAtPath(path, htp_backend_lib_name,
                                    QnnDynSymbolLoader::DynFlag::kRTLD_NOW | QnnDynSymbolLoader::DynFlag::kRTLD_GLOBAL)) {
      MLLM_ERROR("QnnAOTEnv::QnnAOTEnv {} failed.", htp_backend_lib_name);
      exit(1);
    }
  }

  auto qnn_interface_get_providers_func =
      loader(htp_backend_lib_name).func<QnnFuncSymbols::QnnInterfaceGetProvidersFuncType>("QnnInterface_getProviders");

  QnnInterface_t** interface_providers = nullptr;
  uint32_t num_providers = 0;

  MLLM_RT_ASSERT_EQ(qnn_interface_get_providers_func((const QnnInterface_t***)&interface_providers, &num_providers),
                    QNN_SUCCESS);
  MLLM_RT_ASSERT(interface_providers != nullptr);
  MLLM_RT_ASSERT(num_providers != 0);

  MLLM_INFO("QnnAOTEnv::QnnAOTEnv get HTP num_providers: {}", num_providers);

  bool found_valid_interface = false;
  // Get correct provider
  for (size_t provider_id = 0; provider_id < num_providers; provider_id++) {
    if (QNN_API_VERSION_MAJOR == interface_providers[provider_id]->apiVersion.coreApiVersion.major
        && QNN_API_VERSION_MINOR <= interface_providers[provider_id]->apiVersion.coreApiVersion.minor) {
      found_valid_interface = true;
      qnn_htp_func_symbols_.qnn_interface_ = interface_providers[provider_id]->QNN_INTERFACE_VER_NAME;
      break;
    }
  }
  MLLM_RT_ASSERT_EQ(found_valid_interface, true);

  // Check if this HTP Backend has specific property
  if (nullptr != qnn_htp_func_symbols_.qnn_interface_.propertyHasCapability) {
    auto status = qnn_htp_func_symbols_.qnn_interface_.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
    if (status == QNN_PROPERTY_NOT_SUPPORTED) { MLLM_WARN("Device property is not supported"); }

    MLLM_RT_ASSERT(status != QNN_PROPERTY_ERROR_UNKNOWN_KEY);
  }
}

std::shared_ptr<QnnDeviceAndContext> QnnAOTEnv::createContext(const std::string& name) {
  std::shared_ptr<QnnDeviceAndContext> context = std::make_shared<QnnDeviceAndContext>();
  context->name_ = name;

  // 1. create logger and register callback.
  // clang-format off
  MLLM_RT_ASSERT_EQ(qnn_htp_func_symbols_.qnn_interface_.logCreate(__mllmLoggerCallback4QnnLogger,QNN_LOG_LEVEL_VERBOSE, &context->log_), QNN_SUCCESS)
  MLLM_RT_ASSERT_EQ(QNN_BACKEND_NO_ERROR, qnn_htp_func_symbols_.qnn_interface_.backendCreate(context->log_, (const QnnBackend_Config_t**)context->bk_cfg_, &context->bk_handle_))
  // clang-format on

  // 2. Create HTP Device
  // FIXME(wch): we need to model each Hexagon machine with its special device info.
  // clang-format off
  if (nullptr != qnn_htp_func_symbols_.qnn_interface_.deviceCreate) {
    auto status = qnn_htp_func_symbols_.qnn_interface_.deviceCreate(context->log_, nullptr, &context->device_handle_);
    MLLM_RT_ASSERT_EQ(status, QNN_SUCCESS);
  }
  // clang-format on

  // 3. Create Profile
  {
    auto status = qnn_htp_func_symbols_.qnn_interface_.profileCreate(context->bk_handle_, QNN_PROFILE_LEVEL_DETAILED,
                                                                     &context->profile_bk_handle_);
    MLLM_RT_ASSERT_EQ(status, QNN_SUCCESS);
  }

  // 4. Create Context
  {
    auto status = qnn_htp_func_symbols_.qnn_interface_.contextCreate(context->bk_handle_, context->device_handle_,
                                                                     (const QnnContext_Config_t**)&context->qnn_context_config_,
                                                                     &context->qnn_ctx_handle_);
    MLLM_RT_ASSERT_EQ(QNN_CONTEXT_NO_ERROR, status);
  }

  // 5. Register MLLM's Qnn Opset
  // clang-format off
  {
    // FIXME(wch): we need to register our own opset of qnn.
    // struct OpPackageInfo {
    //   std::string path;
    //   std::string interface_provider;
    //   std::string target;
    // };

    // std::vector<OpPackageInfo> op_packages = {
    //     {.path = "libQnnMllmPackageCPU.so", .interface_provider = "MllmPackageInterfaceProvider", .target = "CPU"},
    //     {.path = "libQnnMllmPackageHTP.so", .interface_provider = "MllmPackageInterfaceProvider", .target = "HTP"},
    // };

    // for (const auto& pkg : op_packages) {
    //   if (!qnn_htp_func_symbols_.qnn_interface_.backendRegisterOpPackage) {
    //     MLLM_ERROR_EXIT(ExitCode::kCoreError, "qnn_htp_func_symbols_.qnn_interface_.backendRegisterOpPackage is nullptr.");
    //   }
    //   auto status = qnn_htp_func_symbols_.qnn_interface_.backendRegisterOpPackage(context->bk_handle_, pkg.path.c_str(), pkg.interface_provider.c_str(), pkg.target.c_str());
    //   MLLM_RT_ASSERT_EQ(status, QNN_BACKEND_NO_ERROR);
    //   MLLM_INFO("QNN Registered op package: {}, interface provider: {}, target: {}", pkg.path, pkg.interface_provider, pkg.target);
    // }
  }
  // clang-format on

  MLLM_RT_ASSERT_EQ(contexts_.count(name), 0);
  contexts_[name] = context;
  return context;
}

void QnnAOTEnv::saveContext(const std::string& name, const std::string& path) {
  // TODO
}

void QnnAOTEnv::destroyContext(const std::string& name) {
  // TODO
}

}  // namespace mllm::qnn::aot
