// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <memory>

#include <QnnTypes.h>

#include <QnnContext.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpCommon.h>
#include <HTP/QnnHtpContext.h>

#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/backends/qnn/QNNTypeMacros.hpp"
#include "mllm/compile/ir/linalg/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/QnnTargetMachine.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn::aot {

QnnAOTNodeTensor::QnnAOTNodeTensor(const ir::tensor::TensorValue::ptr_t& v, bool force_static_weight) {
  auto type = parseQnnTensorTypeFromIR(v);
  auto name = v->name();
  auto quant = parseQnnQuantizeParamFromIR(v);

  if (force_static_weight || type == QNN_TENSOR_TYPE_STATIC) {
    tensor_wrapper_ = mllm::qnn::QNNTensorWrapper::createStaticTensor(name, v->tensor_, quant);
  } else {
    tensor_wrapper_ = mllm::qnn::QNNTensorWrapper::create(name, type, v->tensor_, quant);
  }
  setupComplexTensorQuantization(v);  // per-channel and LPBQ cases
}

Qnn_TensorType_t QnnAOTNodeTensor::parseQnnTensorTypeFromIR(const ir::tensor::TensorValue::ptr_t& v) {
  auto type = v->tensor_.memType();
  Qnn_TensorType_t ret_qnn_tensor_type = QNN_TENSOR_TYPE_UNDEFINED;
  switch (type) {
    case kTensorMemTypes_Start: {
      break;
    }

    // For MLLM Frame work to use
    case kNormal: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_NATIVE;
      break;
    }
    case kExtraInput: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_READ;
      break;
    }
    case kExtraOutput: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
      break;
    }
    case kManual: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_READWRITE;
      break;
    }
    case kGlobal: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_STATIC;
      break;
    }

    // Framework need to judge if this tensor is mmap from disk.
    case kParams_Start:
    case kParamsMMAP:
    case kParamsNormal:
    case kParams_End: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_STATIC;
      break;
    }

    // For QNN Backend to use.
    case kQnnAppRead: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_READ;
      break;
    }
    case kQnnAppWrite: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
      break;
    }
    case kQnnAppReadWrite: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_READWRITE;
      break;
    }
    case kTensorMemTypes_End: break;
  }

  // Check Attribute. The Attribute priority is higher than tensor type
  if (v->getAttr("qnn_graph_outputs")) { ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_READ; }
  if (v->getAttr("qnn_graph_inputs")) { ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE; }
  if (v->getAttr("constant")) { ret_qnn_tensor_type = QNN_TENSOR_TYPE_STATIC; }

  return ret_qnn_tensor_type;
}

Qnn_DataType_t QnnAOTNodeTensor::parseQnnDataTypeFromIR(const ir::tensor::TensorValue::ptr_t& v) {
  return mllm::qnn::mllmDataTypeToQnnDataType(v->tensor_.dtype());
}

std::string QnnAOTNodeTensor::parseQnnTensorNameFromIR(const ir::tensor::TensorValue::ptr_t& v) { return v->name(); }

Qnn_QuantizeParams_t QnnAOTNodeTensor::parseQnnQuantizeParamFromIR(const ir::tensor::TensorValue::ptr_t& v) {
  Qnn_QuantizeParams_t ret = QNN_QUANTIZE_PARAMS_INIT;

  MLLM_RT_ASSERT(v->getAttr("quant_recipe"));
  auto quant_spec = v->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_;

  switch (quant_spec->type) {
    case ir::linalg::QuantizationSpecType::kRaw:
    case ir::linalg::QuantizationSpecType::kSymPerChannel:
    case ir::linalg::QuantizationSpecType::kLPBQ: {
      break;
    }
    case ir::linalg::QuantizationSpecType::kAsymPerTensor: {
      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecAsymPerTensor>(quant_spec);
      ret.encodingDefinition = QNN_DEFINITION_DEFINED;
      ret.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
      ret.scaleOffsetEncoding = Qnn_ScaleOffset_t{.scale = cfg->scale.item<float>(), .offset = cfg->zero_point.item<int32_t>()};
      break;
    }
    case ir::linalg::QuantizationSpecType::kSymPerTensor: {
      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecSymPerTensor>(quant_spec);
      ret.encodingDefinition = QNN_DEFINITION_DEFINED;
      ret.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
      ret.scaleOffsetEncoding = Qnn_ScaleOffset_t{.scale = cfg->scale.item<float>(), .offset = 0};
      break;
    }
    default: {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Can't handle kNone type");
    }
  }

  return ret;
}

void QnnAOTNodeTensor::setupComplexTensorQuantization(const ir::tensor::TensorValue::ptr_t& v) {
  MLLM_RT_ASSERT(v->getAttr("quant_recipe"));
  auto quant_spec = v->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_;

  switch (quant_spec->type) {
    case ir::linalg::QuantizationSpecType::kSymPerChannel: {
      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecSymPerChannel>(quant_spec);

      // Prepare data
      auto num_scale_offsets = (uint32_t)v->tensor_.size(cfg->ch_axis);
      std::vector<Qnn_ScaleOffset_t> scale_offsets(num_scale_offsets);
      MLLM_RT_ASSERT_EQ(num_scale_offsets, cfg->scale.size(0));
      MLLM_RT_ASSERT_EQ(cfg->scale.dtype(), kFloat32);
      for (int i = 0; i < num_scale_offsets; ++i) {
        scale_offsets[i].scale = cfg->scale.at<float>({i});
        scale_offsets[i].offset = 0;
      }

      tensor_wrapper_->setScaleOffsetQuantization(scale_offsets, cfg->ch_axis);
      break;
    }
    case ir::linalg::QuantizationSpecType::kLPBQ: {
      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecLPBQ>(quant_spec);

      // Prepare data
      auto num_scale_offsets = (uint32_t)v->tensor_.size(cfg->ch_axis);
      std::vector<Qnn_ScaleOffset_t> scale_offsets(num_scale_offsets);
      MLLM_RT_ASSERT_EQ(num_scale_offsets, cfg->scale_level_1_fp.size(0));
      MLLM_RT_ASSERT_EQ(cfg->scale_level_0_int.dtype(), kUInt8);
      for (int i = 0; i < num_scale_offsets; ++i) {
        scale_offsets[i].scale = cfg->scale_level_1_fp.at<float>({i, 0, 0});
        scale_offsets[i].offset = 0;
      }

      Qnn_BlockwiseExpansion_t blockwise_expansion;
      blockwise_expansion.axis = cfg->ch_axis;
      blockwise_expansion.scaleOffsets = nullptr;  // Will be set by setBlockwiseQuantization
      blockwise_expansion.numBlocksPerAxis = v->tensor_.size(1) / cfg->block_size;
      blockwise_expansion.blockScaleBitwidth = 4;  // 4 bits for uint4 scale
      blockwise_expansion.blockScaleStorageType = QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8;
      blockwise_expansion.blocksScale8 = cfg->scale_level_0_int.ptr<mllm_uint8_t>();

      tensor_wrapper_->setBlockwiseQuantization(blockwise_expansion, scale_offsets);
      break;
    }
    default: break;
  }
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

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::addParamScalar(
    const std::vector<std::shared_ptr<mllm::qnn::QNNParamScalarWrapper>>& params) {
  param_scalar.insert(param_scalar.end(), params.begin(), params.end());
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::emplaceParamScalar(
    const std::shared_ptr<mllm::qnn::QNNParamScalarWrapper>& param) {
  param_scalar.push_back(param);
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::addParamTensor(
    const std::vector<std::shared_ptr<mllm::qnn::QNNParamTensorWrapper>>& params) {
  param_tensor.insert(param_tensor.end(), params.begin(), params.end());
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::emplaceParamTensor(
    const std::shared_ptr<mllm::qnn::QNNParamTensorWrapper>& param) {
  param_tensor.push_back(param);
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::setOpName(const std::string& op_name) {
  op_name_ = op_name;
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::setName(const std::string& name) {
  name_ = name;
  return shared_from_this();
}

std::string QnnAOTNodeOperation::getName() { return name_; }

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::setPackageName(const std::string& package_name) {
  package_name_ = package_name;
  return shared_from_this();
}

QnnAOTGraph::QnnAOTGraph(QNN_INTERFACE_VER_TYPE& qnnInterface, Qnn_BackendHandle_t backendHandle,
                         Qnn_ContextHandle_t contextHandle, const std::string& graphName) {
  qnn_model_ = std::make_shared<mllm::qnn::QNNModel>(qnnInterface, backendHandle);
  qnn_model_->initialize(contextHandle, graphName.c_str(), false);
}

void QnnAOTGraph::addOperation(const QnnAOTNodeOperation::ptr_t& qnn_op) {
  std::vector<std::string> inputNames;
  for (auto& in : qnn_op->inputs) inputNames.push_back(in->getWrapper()->getName());

  std::vector<std::string> outputNames;
  for (auto& out : qnn_op->outputs) outputNames.push_back(out->getWrapper()->getName());

  for (auto& in : qnn_op->inputs) qnn_model_->addTensorWrapper(in->getWrapper());
  for (auto& out : qnn_op->outputs) qnn_model_->addTensorWrapper(out->getWrapper());

  qnn_model_->addNode(QNN_OPCONFIG_VERSION_1, qnn_op->name_, qnn_op->package_name_, qnn_op->op_name_, qnn_op->param_tensor,
                      qnn_op->param_scalar, inputNames, outputNames);

  op_node_.insert({qnn_op->getName(), qnn_op});
}

bool QnnAOTGraph::compile() {
  if (is_compiled_) { return true; }
  bool ret = qnn_model_->finalizeGraph(nullptr, nullptr) == mllm::qnn::MODEL_NO_ERROR;
  is_compiled_ = true;
  return ret;
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
  MLLM_RT_ASSERT_EQ(qnn_htp_func_symbols_.qnn_interface_.logCreate(__mllmQnnLoggerCallback,QNN_LOG_LEVEL_VERBOSE, &context->log_), QNN_SUCCESS)
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

QnnAOTGraph::ptr_t QnnAOTEnv::captureAOTGraph(const std::string& qnn_context_name, const std::string& g_name) {
  if (contexts_.find(qnn_context_name) == contexts_.end()) {
    MLLM_ERROR("Context {} not found", qnn_context_name);
    return nullptr;
  }
  auto& ctx = contexts_[qnn_context_name];
  if (ctx->graphs_.find(g_name) == ctx->graphs_.end()) {
    ctx->graphs_[g_name] =
        std::make_shared<QnnAOTGraph>(qnn_htp_func_symbols_.qnn_interface_, ctx->bk_handle_, ctx->qnn_ctx_handle_, g_name);
  }
  return ctx->graphs_[g_name];
}

void QnnAOTEnv::captureAOTNodeOp(const std::string& qnn_context_name, const std::string& graph_name,
                                 const QnnAOTNodeOperation::ptr_t& op) {
  MLLM_RT_ASSERT_EQ(contexts_.count(qnn_context_name), 1);
  MLLM_RT_ASSERT_EQ(contexts_[qnn_context_name]->graphs_.count(graph_name), 1);
  contexts_[qnn_context_name]->graphs_[graph_name]->addOperation(op);
}

QnnAOTNodeTensor::ptr_t QnnAOTEnv::captureQnnAOTNodeTensor(const std::string& qnn_context_name, const std::string& graph_name,
                                                           const ir::tensor::TensorValue::ptr_t& v, bool force_static_weight) {
  auto __qnn_tensor_name = v->name();

  bool __qnn_enable_static_weight = force_static_weight;

  // Check if this value want static qnn weight. The static qnn weight will be shared through one context in diff graphs!
  if (v->tensor_.memType() == kGlobal || (v->tensor_.memType() <= kParams_End && v->tensor_.memType() >= kParams_Start)
      || v->getAttr("constant")) {
    __qnn_enable_static_weight = true;
  }

  // If static weight is cached, we return it directly.
  if (__qnn_enable_static_weight) {
    MLLM_RT_ASSERT_EQ(contexts_.count(qnn_context_name), 1);
    if (contexts_[qnn_context_name]->static_tensor_.count(__qnn_tensor_name)) {
      return contexts_[qnn_context_name]->static_tensor_[__qnn_tensor_name];
    }
  }

  // If normal weight is cached, we return it directly
  MLLM_RT_ASSERT_EQ(contexts_.count(qnn_context_name), 1);
  MLLM_RT_ASSERT_EQ(contexts_[qnn_context_name]->graphs_.count(graph_name), 1);
  if (contexts_[qnn_context_name]->graphs_[graph_name]->all_tensors_.count(__qnn_tensor_name)) {
    return contexts_[qnn_context_name]->graphs_[graph_name]->all_tensors_[__qnn_tensor_name];
  }

  // There has no Tensor in the cache.
  auto ret = QnnAOTNodeTensor::create(v, __qnn_enable_static_weight);

  return ret;
}

std::shared_ptr<QnnDeviceAndContext> QnnAOTEnv::getContext(const std::string& name) { return contexts_[name]; }

}  // namespace mllm::qnn::aot
