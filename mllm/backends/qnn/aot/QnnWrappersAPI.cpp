// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <QNN/QnnContext.h>
#include <QNN/HTP/QnnHtpDevice.h>
#include <QNN/HTP/QnnHtpCommon.h>
#include <QNN/HTP/QnnHtpContext.h>

#include "mllm/utils/Common.hpp"
#include "mllm/backends/qnn/QNNTypeMacros.hpp"
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

size_t QnnAOTDataTypeSize(Qnn_DataType_t dtype) {
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
      MLLM_ERROR("QnnAOTDataTypeSize: unsupported Qnn_DataType_t {}", static_cast<int>(dtype));
      MLLM_RT_ASSERT(false);
      return 0;
  }
}

QnnAOTParamScalar::QnnAOTParamScalar(const std::string& name, bool value) {
  name_ = name;
  qnn_param_.paramType = QNN_PARAMTYPE_SCALAR;
  qnn_param_.name = name_.c_str();
  qnn_param_.scalarParam.dataType = QNN_DATATYPE_BOOL_8;
  qnn_param_.scalarParam.bool8Value = static_cast<uint8_t>(value);
}

QnnAOTParamScalar::QnnAOTParamScalar(const std::string& name, uint32_t value) {
  name_ = name;
  qnn_param_.paramType = QNN_PARAMTYPE_SCALAR;
  qnn_param_.name = name_.c_str();
  qnn_param_.scalarParam.dataType = QNN_DATATYPE_UINT_32;
  qnn_param_.scalarParam.uint32Value = value;
}

QnnAOTParamScalar::QnnAOTParamScalar(const std::string& name, float value) {
  name_ = name;
  qnn_param_.paramType = QNN_PARAMTYPE_SCALAR;
  qnn_param_.name = name_.c_str();
  qnn_param_.scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
  qnn_param_.scalarParam.floatValue = value;
}

Qnn_Param_t* QnnAOTParamScalar::getQnnParam() { return &(qnn_param_); }

QnnAOTParamTensor::QnnAOTParamTensor(const std::string& param_name, const std::string& tensor_name, Qnn_DataType_t data_type,
                                     const std::vector<uint32_t>& dimensions) {
  param_name_ = param_name;
  tensor_name_ = tensor_name;
  dimensions_ = dimensions;
  // Fix parameters.
  qnn_param_.paramType = QNN_PARAMTYPE_TENSOR;
  qnn_param_.tensorParam.version = QNN_TENSOR_VERSION_2;
  qnn_param_.tensorParam.v2 = QNN_TENSOR_V2_INIT;
  qnn_param_.tensorParam.v2.type = QNN_TENSOR_TYPE_STATIC;
  qnn_param_.tensorParam.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  qnn_param_.tensorParam.v2.quantizeParams = Qnn_QuantizeParams_t{
      QNN_DEFINITION_UNDEFINED, QNN_QUANTIZATION_ENCODING_UNDEFINED, {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}};
  qnn_param_.tensorParam.v2.memType = QNN_TENSORMEMTYPE_RAW;
  // Custom parameters.
  qnn_param_.name = param_name_.c_str();
  qnn_param_.tensorParam.v2.name = tensor_name_.c_str();
  qnn_param_.tensorParam.v2.dataType = data_type;
  qnn_param_.tensorParam.v2.rank = dimensions_.size();
  qnn_param_.tensorParam.v2.dimensions = dimensions_.data();
  qnn_param_.tensorParam.v2.clientBuf = {.data = nullptr, .dataSize = 0};
}

QnnAOTParamTensor::~QnnAOTParamTensor() {
  auto data = QNN_TENSOR_GET_CLIENT_BUF(qnn_param_.tensorParam).data;
  MLLM_RT_ASSERT(data != nullptr);
  if (data) { free(data); }
}

void* QnnAOTParamTensor::alloc() {
  uint32_t data_size = QnnAOTDataTypeSize(QNN_TENSOR_GET_DATA_TYPE(qnn_param_.tensorParam));
  for (int i = 0; i < QNN_TENSOR_GET_RANK(qnn_param_.tensorParam); i++) {
    data_size *= qnn_param_.tensorParam.v2.dimensions[i];
  }
  Qnn_ClientBuffer_t clientBuffer = {.data = malloc(data_size), .dataSize = data_size};
  QNN_TENSOR_SET_CLIENT_BUF(qnn_param_.tensorParam, clientBuffer);
  MLLM_RT_ASSERT(QNN_TENSOR_GET_CLIENT_BUF(qnn_param_.tensorParam).data != nullptr);
  return QNN_TENSOR_GET_CLIENT_BUF(qnn_param_.tensorParam).data;
}

Qnn_Param_t* QnnAOTParamTensor::getQnnParam() { return &qnn_param_; }

Qnn_Tensor_t* QnnAOTParamTensor::getQnnTensor() { return &qnn_param_.tensorParam; }

QnnAOTNodeTensor::QnnAOTNodeTensor(const ir::tensor::TensorValue::ptr_t& v) {
  tensor_ir_ = v;
  // TODO
}

Qnn_TensorType_t QnnAOTNodeTensor::parseQnnTensorTypeFromIR() {
  // TODO
}

Qnn_DataType_t QnnAOTNodeTensor::parseQnnDataTypeFromIR() {
  // TODO
}

std::string QnnAOTNodeTensor::parseQnnTensorNameFromIR() {
  // TODO
}

Qnn_QuantizeParams_t QnnAOTNodeTensor::parseQnnQuantizeParamFromIR() {
  // TODO
}

// QnnAOTNodeOperation implementations
QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::addInputs(const std::vector<QnnAOTNodeTensor::ptr_t>& ins) {
  inputs.insert(inputs.end(), ins.begin(), ins.end());
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::addOutputs(const std::vector<QnnAOTNodeTensor::ptr_t>& ous) {
  outputs.insert(outputs.end(), ous.begin(), ous.end());
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::emplaceInput(const QnnAOTNodeTensor::ptr_t& input) {
  inputs.push_back(input);
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::emplaceOutput(const QnnAOTNodeTensor::ptr_t& output) {
  outputs.push_back(output);
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::addParamScalar(const std::vector<QnnAOTParamScalar::ptr_t>& params) {
  param_scalar.insert(param_scalar.end(), params.begin(), params.end());
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::emplaceParamScalar(const QnnAOTParamScalar::ptr_t& param) {
  param_scalar.push_back(param);
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::addParamTensor(const std::vector<QnnAOTParamTensor::ptr_t>& params) {
  param_tensor.insert(param_tensor.end(), params.begin(), params.end());
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::emplaceParamTensor(const QnnAOTParamTensor::ptr_t& param) {
  param_tensor.push_back(param);
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::setOpName(const std::string& op_name) {
  op_name_ = op_name;
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::setPackageName(const std::string& package_name) {
  package_name_ = package_name;
  return shared_from_this();
}

const std::vector<std::string> QnnDynSymbolLoader::possible_qnn_dyn_lib_paths_{
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

QnnAOTEnv::QnnAOTEnv(const QcomTargetMachine& target_machine) : target_machine_(target_machine) { _setup(); }

QnnAOTEnv::QnnAOTEnv(const std::string& lib_path, const QcomTargetMachine& target_machine) : target_machine_(target_machine) {
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

  // Try to config this target machine
  {
    auto device_custom_config = createDecideCustomConfigInfo();
    QnnHtpDevice_CustomConfig_t* p_custom_config = nullptr;

    switch (target_machine_.soc_htp_security_pd_session) {
      case QcomSecurityPDSession::kHtpSignedPd: {
        p_custom_config = (QnnHtpDevice_CustomConfig_t*)malloc(sizeof(QnnHtpDevice_CustomConfig_t));
        unreachable_handle_.push_back(p_custom_config);
        p_custom_config->option = QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD;
        p_custom_config->useSignedProcessDomain.useSignedProcessDomain = true;
        p_custom_config->useSignedProcessDomain.deviceId = 0;
        device_custom_config.push_back(static_cast<QnnDevice_CustomConfig_t>(p_custom_config));
        break;
      }
      case QcomSecurityPDSession::kHtpUnsignedPd:
      default: break;
    }

    const std::vector<QnnDevice_PlatformInfo_t*> device_platform_info = createDevicePlatformInfo();
    uint32_t num_custom_configs = device_platform_info.size() + device_custom_config.size();
    target_machine_qnn_config_.resize(num_custom_configs);

    for (std::size_t i = 0; i < device_custom_config.size(); ++i) {
      target_machine_qnn_config_[i].option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
      target_machine_qnn_config_[i].customConfig = device_custom_config[i];
      target_machine_qnn_config_ptrs_.push_back(&target_machine_qnn_config_[i]);
    }

    if (!device_platform_info.empty()) {
      // The length of platform info can only be 1.
      MLLM_RT_ASSERT_EQ(device_platform_info.size(), 1u);
      target_machine_qnn_config_[device_custom_config.size()].option = QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO;
      target_machine_qnn_config_[device_custom_config.size()].hardwareInfo = device_platform_info.back();
      target_machine_qnn_config_ptrs_.push_back(&target_machine_qnn_config_[device_custom_config.size()]);
    }

    // null terminated
    target_machine_qnn_config_ptrs_.push_back(nullptr);
  }
}

std::shared_ptr<QnnDeviceAndContext> QnnAOTEnv::createContext(const std::string& name, bool weights_sharing) {
  std::shared_ptr<QnnDeviceAndContext> context = std::make_shared<QnnDeviceAndContext>();
  context->name_ = name;

  // 1. create logger and register callback.
  // clang-format off
  MLLM_RT_ASSERT_EQ(qnn_htp_func_symbols_.qnn_interface_.logCreate(__mllmLoggerCallback4QnnLogger,QNN_LOG_LEVEL_VERBOSE, &context->log_), QNN_SUCCESS)
  MLLM_RT_ASSERT_EQ(QNN_BACKEND_NO_ERROR, qnn_htp_func_symbols_.qnn_interface_.backendCreate(context->log_, (const QnnBackend_Config_t**)context->bk_cfg_, &context->bk_handle_))
  // clang-format on

  // 2. Create HTP Device
  // clang-format off
  if (nullptr != qnn_htp_func_symbols_.qnn_interface_.deviceCreate) {
    auto status = qnn_htp_func_symbols_.qnn_interface_.deviceCreate(context->log_, target_machine_qnn_config_ptrs_.data(), &context->device_handle_);
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
    auto cfgs = createContextCustomConfig(weights_sharing);
    if (cfgs.size()) {
      context->qnn_context_config_ = (QnnContext_Config_t**)malloc(sizeof(QnnContext_Config_t*) * (cfgs.size() + 1));
      unreachable_handle_.emplace_back(context->qnn_context_config_);
    }
    for (int i = 0; i < cfgs.size(); ++i) {
      context->qnn_context_config_[i] = (QnnContext_Config_t*)malloc(sizeof(QnnContext_Config_t));
      context->qnn_context_config_[i]->option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
      context->qnn_context_config_[i]->customConfig = cfgs[i];
      unreachable_handle_.emplace_back(context->qnn_context_config_[i]);
    }
    if (cfgs.size()) { context->qnn_context_config_[cfgs.size()] = nullptr; }
    auto status = qnn_htp_func_symbols_.qnn_interface_.contextCreate(context->bk_handle_, context->device_handle_,
                                                                     (const QnnContext_Config_t**)context->qnn_context_config_,
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

std::vector<QnnDevice_PlatformInfo_t*> QnnAOTEnv::createDevicePlatformInfo() {
  std::vector<QnnDevice_PlatformInfo_t*> ret;
  QnnDevice_PlatformInfo_t* p_platform_info = nullptr;
  QnnDevice_HardwareDeviceInfo_t* p_hw_device_info = nullptr;
  QnnHtpDevice_DeviceInfoExtension_t* p_device_info_extension = nullptr;
  QnnDevice_CoreInfo_t* p_core_info = nullptr;

  p_platform_info = (QnnDevice_PlatformInfo_t*)malloc(sizeof(QnnDevice_PlatformInfo_t));
  unreachable_handle_.push_back(p_platform_info);
  p_platform_info->version = QNN_DEVICE_PLATFORM_INFO_VERSION_1;
  p_platform_info->v1.numHwDevices = 1;

  p_hw_device_info = (QnnDevice_HardwareDeviceInfo_t*)malloc(sizeof(QnnDevice_HardwareDeviceInfo_t));
  unreachable_handle_.push_back(p_hw_device_info);
  p_hw_device_info->version = QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1;
  p_hw_device_info->v1.deviceId = 0;
  p_hw_device_info->v1.deviceType = 0;
  p_hw_device_info->v1.numCores = 1;

  p_device_info_extension = (QnnHtpDevice_DeviceInfoExtension_t*)malloc(sizeof(QnnHtpDevice_DeviceInfoExtension_t));
  unreachable_handle_.push_back(p_device_info_extension);
  // clang-format off
  p_device_info_extension->devType = QNN_HTP_DEVICE_TYPE_ON_CHIP;
  p_device_info_extension->onChipDevice.vtcmSize = target_machine_.soc_htp_vtcm_total_memory_size;  // in MB
  p_device_info_extension->onChipDevice.signedPdSupport = target_machine_.soc_htp_security_pd_session == QcomSecurityPDSession::kHtpSignedPd;
  p_device_info_extension->onChipDevice.socModel = static_cast<uint32_t>(target_machine_.soc_htp_chipset);
  p_device_info_extension->onChipDevice.arch = static_cast<QnnHtpDevice_Arch_t>(target_machine_.soc_htp_arch);
  p_device_info_extension->onChipDevice.dlbcSupport = true;
  p_hw_device_info->v1.deviceInfoExtension = p_device_info_extension;
  // clang-format on

  p_core_info = (QnnDevice_CoreInfo_t*)malloc(sizeof(QnnDevice_CoreInfo_t));
  unreachable_handle_.push_back(p_core_info);
  p_core_info->version = QNN_DEVICE_CORE_INFO_VERSION_1;
  p_core_info->v1.coreId = 0;
  p_core_info->v1.coreType = 0;
  p_core_info->v1.coreInfoExtension = nullptr;
  p_hw_device_info->v1.cores = p_core_info;

  p_platform_info->v1.hwDevices = p_hw_device_info;
  ret.push_back(p_platform_info);

  return ret;
}

std::vector<QnnDevice_CustomConfig_t> QnnAOTEnv::createDecideCustomConfigInfo() {
  std::vector<QnnDevice_CustomConfig_t> ret;

  QnnHtpDevice_CustomConfig_t* p_custom_config = (QnnHtpDevice_CustomConfig_t*)malloc(sizeof(QnnHtpDevice_CustomConfig_t));
  unreachable_handle_.push_back(p_custom_config);
  p_custom_config->option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
  p_custom_config->socModel = static_cast<uint32_t>(target_machine_.soc_htp_chipset);
  ret.push_back(static_cast<QnnDevice_CustomConfig_t>(p_custom_config));

  return ret;
}

std::vector<QnnContext_CustomConfig_t> QnnAOTEnv::createContextCustomConfig(bool weights_sharing) {
  std::vector<QnnContext_CustomConfig_t> ret;
  QnnHtpContext_CustomConfig_t* p_custom_config = nullptr;

  if (weights_sharing) {
    p_custom_config = (QnnHtpContext_CustomConfig_t*)malloc(sizeof(QnnHtpContext_CustomConfig_t));
    unreachable_handle_.push_back(p_custom_config);
    p_custom_config->option = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
    p_custom_config->weightSharingEnabled = true;
    ret.push_back(static_cast<QnnContext_CustomConfig_t>(p_custom_config));
  }

  return ret;
}

}  // namespace mllm::qnn::aot
