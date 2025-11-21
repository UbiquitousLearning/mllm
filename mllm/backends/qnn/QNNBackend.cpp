#include "QNNBackend.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <memory>
#include "QNNUtils.hpp"
#include "QnnLog.h"
#include "mllm/backends/qnn/QNNAllocator.hpp"
#include "mllm/backends/qnn/op/QNNCastTypeOp.hpp"
#include "mllm/backends/qnn/op/QNNElewiseOp.hpp"
#include "mllm/backends/qnn/op/QNNEmbeddingOp.hpp"
#include "mllm/backends/qnn/op/QNNGraphOp.hpp"
#include "mllm/backends/qnn/op/QNNLinearOp.hpp"
#include "mllm/backends/qnn/op/QNNParamOp.hpp"
#include "mllm/backends/qnn/op/QNNRMSNormOp.hpp"
#include "mllm/backends/qnn/op/QNNSiLUOp.hpp"
#include "mllm/backends/qnn/op/QNNTransposeOp.hpp"
#include "mllm/backends/qnn/op/QNNViewOp.hpp"
#include "mllm/backends/qnn/op/QNNX2XOp.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn {

QNNBackend::QNNBackend() : Backend(kQNN, createQNNAllocator()) {
  // register ops
  regOpFactory<QNNAddOpFactory, QNNMulOpFactory, QNNGraphBeginOpFactory, QNNGraphEndOpFactory, QNNLinearOpFactory,
               QNNViewOpFactory, QNNRMSNormOpFactory, QNNTransposeOpFactory, QNNX2XOpFactory, QNNCastTypeOpFactory,
               QNNParamOpFactory, QNNSiLUOpFactory, QNNEmbeddingOpFactory>();

  QnnLog_Level_t qnnLogLevel = QNN_LOG_LEVEL_ERROR;  // default QNN log level
  profilingLevel_ = ProfilingLevel::OFF;
  debug_ = false;  // when set true, NATIVE tensor will be regared as APP_READ tensor

  loadQNNSymbol();
  loadQNNSystemSymbol();

  runtime_ = QNNRuntime::create(profilingLevel_, qnnLogLevel);
  if (!runtime_) { MLLM_ERROR_EXIT(1, "Failed to create QNN Runtime"); }

  // check QNN capability, detect QNN features for future use
  char* backendBuildId{nullptr};
  if (QNN_SUCCESS != runtime_->qnnInterface.backendGetBuildId((const char**)&backendBuildId)) {
    MLLM_ERROR("Unable to get build Id from the backend.");
  }
  MLLM_INFO("QNN Backend Build Id: {}", backendBuildId == nullptr ? "" : backendBuildId);
  if (runtime_->qnnInterface.propertyHasCapability(QNN_PROPERTY_TENSOR_SUPPORT_SPARSITY) == QNN_PROPERTY_SUPPORTED) {
    MLLM_INFO("QNN backend supports tensor sparsity");
  }
  if (runtime_->qnnInterface.propertyHasCapability(QNN_PROPERTY_TENSOR_SUPPORT_DYNAMIC_DIMENSIONS) == QNN_PROPERTY_SUPPORTED) {
    MLLM_INFO("QNN backend supports dynamic dimensions");
  }
  if (runtime_->qnnInterface.propertyHasCapability(QNN_PROPERTY_GRAPH_SUPPORT_EARLY_TERMINATION) == QNN_PROPERTY_SUPPORTED) {
    MLLM_INFO("QNN backend supports early termination");
  }

  bool contextStatus = false;
  // check if the qnn_context.bin file exists
  if (!std::filesystem::exists("qnn_context.bin")) {
    contextStatus = runtime_->createContext(context_, nullptr);
  } else {
    contextStatus = runtime_->retrieveContext(context_, qnnModels_, nullptr);

    // fill qnnModelIndexMap_ info according to qnnModels_
    for (size_t i = 0; i < qnnModels_.size(); i++) {
      auto graphName = qnnModels_[i]->getQnnGraphName();
      qnnModelIndexMap_.insert(std::make_pair(graphName, i));
    }
  }
  if (!contextStatus) { MLLM_ERROR_EXIT(1, "Failed to create QNN context"); }

  // init QNN Allocator
  static_pointer_cast<QNNAllocator>(allocator_)->setQNNPointer(runtime_->qnnInterface, context_);

  // set performance parameters for better performance on HTP
  perf_ = QNNPerf::create(&runtime_->qnnInterface);
  perf_->setPowerConfigBurst();
  perf_->setRpcLatencyAndPolling();
}

QNNPerf::QNNPerf(const QNN_INTERFACE_VER_TYPE* qnnInterface) {
  assert(qnnInterface != nullptr);
  mQnnInterface = qnnInterface;

  QnnDevice_Infrastructure_t deviceInfra = nullptr;
  CALL_QNN(mQnnInterface->deviceGetInfrastructure(&deviceInfra));
  QnnHtpDevice_Infrastructure_t* htpInfra = static_cast<QnnHtpDevice_Infrastructure_t*>(deviceInfra);
  mPerfInfra = htpInfra->perfInfra;

  uint32_t deviceId = 0;
  uint32_t coreId = 0;
  CALL_QNN(mPerfInfra.createPowerConfigId(deviceId, coreId, &mPowerConfigId));

  mPowerConfigBurst = {
      .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
      .dcvsV3Config =
          {
              .contextId = mPowerConfigId,  // use the power config id created
              .setDcvsEnable = 1,
              .dcvsEnable = 0,  // 1- To enable Dcvs and consider dcvs power mode, 0- To disable dcvs
              .powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
              .setSleepLatency = 1,  // True to consider Latency parameter otherwise False
              .sleepLatency = 40,    // set dsp sleep latency ranges 10-65535 micro sec, refer hexagon sdk
              .setSleepDisable = 1,  // True to consider sleep disable/enable parameter otherwise False
              .sleepDisable = 1,     // True to disable sleep, False to re-enable sleep
              .setBusParams = 1,     // True to consider Bus parameter otherwise False
              .busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .setCoreParams = 1,  // True to consider Core parameter otherwise False
              .coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
              .coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
          },
  };

  mPowerConfigBalanced = {
      .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
      .dcvsV3Config =
          {
              .contextId = mPowerConfigId,  // use the power config id created
              .setDcvsEnable = 1,
              .dcvsEnable = 1,  // 1- To enable Dcvs and consider dcvs power mode, 0- To disable dcvs
              .powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
              .setSleepLatency = 1,  // True to consider Latency parameter otherwise False
              .sleepLatency = 1000,  // set dsp sleep latency ranges 10-65535 micro sec, refer hexagon sdk
              .setSleepDisable = 1,  // True to consider sleep disable/enable parameter otherwise False
              .sleepDisable = 0,     // True to disable sleep, False to re-enable sleep
              .setBusParams = 1,     // True to consider Bus parameter otherwise False
              .busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO,
              .busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO,
              .busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO,
              .setCoreParams = 1,  // True to consider Core parameter otherwise False
              .coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO,
              .coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO,
              .coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO,
          },
  };
}

// destory power config
QNNPerf::~QNNPerf() { CALL_QNN(mPerfInfra.destroyPowerConfigId(mPowerConfigId)); }

void QNNPerf::setRpcLatencyAndPolling() {
  // set RPC Control Latency
  QnnHtpPerfInfrastructure_PowerConfig_t rpcControlLatency;  // refer QnnHtpPerfInfrastructure.h
  ::memset(&rpcControlLatency, 0, sizeof(rpcControlLatency));
  rpcControlLatency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
  rpcControlLatency.rpcControlLatencyConfig = 100;  // use rpc control latency recommended 100 us, refer hexagon sdk
  const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs1[] = {&rpcControlLatency, nullptr};

  CALL_QNN(mPerfInfra.setPowerConfig(mPowerConfigId, powerConfigs1));  // set RPC latency config on power config ID created

  // set RPC Polling
  QnnHtpPerfInfrastructure_PowerConfig_t rpcPollingTime;  // refer QnnHtpPerfInfrastructure.h
  ::memset(&rpcPollingTime, 0, sizeof(rpcPollingTime));
  rpcPollingTime.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
  rpcPollingTime.rpcPollingTimeConfig = 9999;  // use rpc polling time recommended 0-10000 us
  const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs2[] = {&rpcPollingTime, nullptr};

  CALL_QNN(mPerfInfra.setPowerConfig(mPowerConfigId, powerConfigs2));  // set RPC polling config on power config ID created
}

void QNNPerf::setPowerConfigBurst() {
  const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs[] = {&mPowerConfigBurst, nullptr};
  CALL_QNN(mPerfInfra.setPowerConfig(mPowerConfigId, powerConfigs));
}

void QNNPerf::setPowerConfigBalanced() {
  const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs[] = {&mPowerConfigBalanced, nullptr};
  CALL_QNN(mPerfInfra.setPowerConfig(mPowerConfigId, powerConfigs));
}

QNNRuntime::~QNNRuntime() {
  // Free Profile
  if (profileHandle != nullptr) { CALL_QNN(qnnInterface.profileFree(profileHandle)); }

  // Free Device
  CALL_QNN(qnnInterface.deviceFree(deviceHandle));

  // Free Backend
  CALL_QNN(qnnInterface.backendFree(backendHandle));

  // Free Log
  CALL_QNN(qnnInterface.logFree(logHandle));
}

QNNRuntime* QNNRuntime::initRuntime(ProfilingLevel profilingLevel, QnnLog_Level_t qnnLogLevel) {
  // Create Interface
  QNN_INTERFACE_VER_TYPE qnnInterface{};
  {
    QnnInterface_t** interfaceProviders = nullptr;
    uint32_t numProviders = 0;
    if (QnnInterface_getProviders((const QnnInterface_t***)&interfaceProviders, &numProviders) != QNN_SUCCESS) {
      MLLM_ERROR("Failed to get QNN interface providers.");
      return nullptr;
    }
    if (interfaceProviders == nullptr) {
      MLLM_ERROR("Failed to get interface providers: null interface providers received.");
      return nullptr;
    }
    if (numProviders == 0) {
      MLLM_ERROR("Failed to get interface providers: 0 interface providers.");
      return nullptr;
    }
    bool foundValidInterface = false;
    for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
      if (QNN_API_VERSION_MAJOR == interfaceProviders[pIdx]->apiVersion.coreApiVersion.major
          && QNN_API_VERSION_MINOR <= interfaceProviders[pIdx]->apiVersion.coreApiVersion.minor) {
        foundValidInterface = true;
        qnnInterface = interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
        break;
      }
    }
    if (!foundValidInterface) {
      MLLM_ERROR("Failed to find a valid QNN interface provider.");
      return nullptr;
    }
  }

  // Create Log
  Qnn_LogHandle_t logHandle = nullptr;
  {
    QnnLog_Callback_t logCallback = &__mllmQnnLoggerCallback;
    if ((QNN_GET_ERROR_CODE(qnnInterface.logCreate(logCallback, qnnLogLevel, &logHandle)) != QNN_SUCCESS)
        || (logHandle == nullptr)) {
      MLLM_ERROR("Failed to initialize logging in the backend.");
      return nullptr;
    }
  }

  // Create Backend
  Qnn_BackendHandle_t backendHandle = nullptr;
  {
    const QnnBackend_Config_t** backendConfig = nullptr;
    if ((QNN_GET_ERROR_CODE(qnnInterface.backendCreate(logHandle, backendConfig, &backendHandle)) != QNN_SUCCESS)
        || (backendHandle == nullptr)) {
      MLLM_ERROR("Failed to create the backend.");
      return nullptr;
    }
  }

  // Create Device
  Qnn_DeviceHandle_t deviceHandle = nullptr;
  {
    // Check whether the device API is supported.
    if (nullptr != qnnInterface.propertyHasCapability) {
      auto qnnStatus = qnnInterface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
      if (QNN_PROPERTY_NOT_SUPPORTED == qnnStatus) {
        MLLM_WARN("Device property is not supported");
        return nullptr;
      }
      if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnStatus) {
        MLLM_ERROR("Device property is not known to backend");
        return nullptr;
      }
    }
  }

  // Initialize Profiling
  Qnn_ProfileHandle_t profileHandle = nullptr;
  {
    if (ProfilingLevel::OFF != profilingLevel) {
      MLLM_INFO("Profiling turned on; level = {}", (int)profilingLevel);
      if (ProfilingLevel::BASIC == profilingLevel) {
        MLLM_INFO("Basic profiling requested. Creating Qnn Profile object.");
        if (QNN_PROFILE_NO_ERROR != qnnInterface.profileCreate(backendHandle, QNN_PROFILE_LEVEL_BASIC, &profileHandle)) {
          MLLM_WARN("Unable to create profile handle in the backend.");
          return nullptr;
        }
      } else if (ProfilingLevel::DETAILED == profilingLevel) {
        MLLM_INFO("Detailed profiling requested. Creating Qnn Profile object.");
        if (QNN_PROFILE_NO_ERROR != qnnInterface.profileCreate(backendHandle, QNN_PROFILE_LEVEL_DETAILED, &profileHandle)) {
          MLLM_ERROR("Unable to create profile handle in the backend.");
          return nullptr;
        }
      }
    }
  }

  // Register Custom OpPackages
  {
    struct OpPackageInfo {
      std::string path;
      std::string interfaceProvider;
      std::string target;
    };

    std::vector<OpPackageInfo> opPackages = {
        {.path = "libQnnLLaMAPackage_CPU.so", .interfaceProvider = "LLaMAPackageInterfaceProvider", .target = "CPU"},
        {.path = "libQnnLLaMAPackage_HTP.so", .interfaceProvider = "LLaMAPackageInterfaceProvider", .target = "HTP"}};

    for (const auto& pkg : opPackages) {
      if (!qnnInterface.backendRegisterOpPackage) {
        MLLM_ERROR("backendRegisterOpPackageFnHandle is nullptr.");
        return nullptr;
      }
      if (QNN_BACKEND_NO_ERROR
          != qnnInterface.backendRegisterOpPackage(backendHandle, pkg.path.c_str(), pkg.interfaceProvider.c_str(),
                                                   pkg.target.c_str())) {
        MLLM_ERROR("Could not register Op Package: {} and interface provider: {}", pkg.path.c_str(),
                   pkg.interfaceProvider.c_str());
        return nullptr;
      }
      MLLM_INFO("Registered Op Package: {} and interface provider: {}", pkg.path.c_str(), pkg.interfaceProvider.c_str());
    }
  }

  // Create QNN System Interface
  QNN_SYSTEM_INTERFACE_VER_TYPE qnnSystemInterface;
  {
    QnnSystemInterface_t** systemInterfaceProviders{nullptr};
    uint32_t numProviders{0};
    if (QNN_SUCCESS
        != QnnSystemInterface_getProviders((const QnnSystemInterface_t***)&systemInterfaceProviders, &numProviders)) {
      MLLM_ERROR("Failed to get system interface providers.");
      return nullptr;
    }
    if (0 == numProviders) {
      MLLM_ERROR("Failed to get interface providers: 0 interface providers.");
      return nullptr;
    }
    bool foundValidSystemInterface = false;
    for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
      foundValidSystemInterface = true;
      if (QNN_SYSTEM_API_VERSION_MAJOR == systemInterfaceProviders[pIdx]->systemApiVersion.major
          && QNN_SYSTEM_API_VERSION_MINOR <= systemInterfaceProviders[pIdx]->systemApiVersion.minor) {
        qnnSystemInterface = systemInterfaceProviders[pIdx]->QNN_SYSTEM_INTERFACE_VER_NAME;
        break;
      }
    }
    if (!foundValidSystemInterface) {
      MLLM_ERROR("Unable to find a valid system interface.");
      return nullptr;
    }
  }

  return new QNNRuntime(qnnInterface, qnnSystemInterface, logHandle, backendHandle, deviceHandle, profileHandle);
}

bool QNNRuntime::createContext(Qnn_ContextHandle_t& context, QnnContext_Config_t** contextConfig) {
  if (QNN_CONTEXT_NO_ERROR
      != qnnInterface.contextCreate(backendHandle, deviceHandle, (const QnnContext_Config_t**)&contextConfig, &context)) {
    MLLM_ERROR("Could not create context");
    return false;
  }
  return true;
}

bool QNNRuntime::retrieveContext(Qnn_ContextHandle_t& context, std::vector<std::shared_ptr<QNNModel>>& qnnModels,
                                 QnnContext_Config_t** contextConfig) {
  // Read the binary from qnn_context.bin and get the size in byte
  std::ifstream file(QNN_Context_File, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  auto binaryBuffer = std::make_unique<uint8_t[]>(size);

  file.read(reinterpret_cast<char*>(binaryBuffer.get()), size);
  file.close();

  // inspect binary info
  QnnSystemContext_Handle_t sysCtxHandle{nullptr};
  if (QNN_SUCCESS != qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
    MLLM_ERROR("Could not create system handle.");
    return false;
  }
  const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
  Qnn_ContextBinarySize_t binaryInfoSize{0};
  if (QNN_SUCCESS
      != qnnSystemInterface.systemContextGetBinaryInfo(sysCtxHandle, static_cast<void*>(binaryBuffer.get()), size, &binaryInfo,
                                                       &binaryInfoSize)) {
    MLLM_ERROR("Failed to get context binary info");
    return false;
  }

  // Extract graph metadata to create QNNModels instead of GraphInfo_t
  GraphInfo_t** tmpGraphsInfo = nullptr;
  uint32_t graphNum;
  // fill GraphInfo_t based on binary info - temporarily needed for tensor extraction
  if (!copyMetadataToGraphsInfo(binaryInfo, tmpGraphsInfo, graphNum)) {
    MLLM_ERROR("Failed to copy metadata.");
    return false;
  }
  qnnSystemInterface.systemContextFree(sysCtxHandle);
  sysCtxHandle = nullptr;

  // Create context from binary
  Qnn_ContextBinarySize_t writtenSize = 0;
  qnnInterface.contextCreateFromBinary(backendHandle, deviceHandle, (const QnnContext_Config_t**)contextConfig,
                                       binaryBuffer.get(), size, &context, profileHandle);

  // Create QNNModels for each graph and initialize from context
  qnnModels.clear();
  qnnModels.reserve(graphNum);

  for (uint32_t i = 0; i < graphNum; ++i) {
    GraphInfo_t* graphInfo = tmpGraphsInfo[i];

    // Retrieve the graph handle
    Qnn_GraphHandle_t graph = nullptr;
    if (QNN_SUCCESS != qnnInterface.graphRetrieve(context, graphInfo->graphName, &graph)) {
      MLLM_ERROR("Unable to retrieve graph handle for graph: {}", graphInfo->graphName);
      return false;
    }

    // Create QNNModel and initialize from context
    auto qnnModel = std::make_shared<QNNModel>(qnnInterface, backendHandle);
    ModelError_t err =
        qnnModel->initializeFromContext(context, graphInfo->graphName, graph, graphInfo->inputTensors,
                                        graphInfo->numInputTensors, graphInfo->outputTensors, graphInfo->numOutputTensors);

    if (err != MODEL_NO_ERROR) {
      MLLM_ERROR("Failed to initialize QNNModel from context for graph: {} with error: {}", graphInfo->graphName,
                 static_cast<int>(err));
      return false;
    }

    qnnModels.push_back(qnnModel);
    MLLM_INFO("Successfully created QNNModel for graph: {}", graphInfo->graphName);
  }

  // Clean up temporary GraphInfo_t structures
  for (uint32_t i = 0; i < graphNum; ++i) {
    if (tmpGraphsInfo[i]) {
      if (tmpGraphsInfo[i]->graphName) { free(tmpGraphsInfo[i]->graphName); }
      freeQnnTensors(tmpGraphsInfo[i]->inputTensors, tmpGraphsInfo[i]->numInputTensors);
      freeQnnTensors(tmpGraphsInfo[i]->outputTensors, tmpGraphsInfo[i]->numOutputTensors);
      free(tmpGraphsInfo[i]);
    }
  }
  free(tmpGraphsInfo);

  MLLM_INFO("QNN context retrieved from qnn_context.bin with {} QNNModels", graphNum);
  return true;
}

std::shared_ptr<QNNModel> QNNBackend::createQnnGraph(const std::string& graphName) {
  // If the graph already exists, return the existing model
  if (qnnModelIndexMap_.find(graphName) != qnnModelIndexMap_.end()) {
    currentQnnModelIndex_ = qnnModelIndexMap_[graphName];
    return qnnModels_[currentQnnModelIndex_];
  }

  // Create a new QNNModel
  currentQnnModelIndex_ = static_cast<int>(qnnModels_.size());
  qnnModelIndexMap_.insert(std::make_pair(graphName, currentQnnModelIndex_));

  auto qnnModel = std::make_shared<QNNModel>(runtime_->qnnInterface, runtime_->backendHandle);
  qnnModels_.push_back(qnnModel);

  // Initialize QNN graph info with basic configs
  const QnnGraph_Config_t* graphConfigList[] = {nullptr};

  ModelError_t err = MODEL_NO_ERROR;
  if ((err = qnnModel->initialize(context_, graphName.c_str(), debug_, 1, graphConfigList)) != MODEL_NO_ERROR) {
    MLLM_ERROR("QNN graph initialization failed for graph: {} with error code: {}", graphName, static_cast<int>(err));
    qnnModels_.pop_back();
    qnnModelIndexMap_.erase(graphName);
    return nullptr;
  }

  return qnnModel;
}

void QNNBackend::graphAddNode(const std::string& graphName, const std::string& nodeName, const std::string& nodeType,
                              const std::vector<std::string>& inputTensorNames,
                              const std::vector<std::string>& outputTensorNames,
                              const std::vector<std::shared_ptr<QNNParamTensorWrapper>>& tensorParams,
                              const std::vector<std::shared_ptr<QNNParamScalarWrapper>>& scalarParams,
                              const std::string& packageName) {
  auto it = qnnModelIndexMap_.find(graphName);
  if (it == qnnModelIndexMap_.end()) {
    MLLM_ERROR("Graph {} not found for adding node", graphName);
    return;
  }

  int modelIndex = it->second;
  auto& qnnModel = qnnModels_[modelIndex];

  if (qnnModel->isGraphFinalized()) { return; }

  // Add node to the model
  ModelError_t err = qnnModel->addNode(QNN_OPCONFIG_VERSION_1, nodeName, packageName, nodeType, tensorParams, scalarParams,
                                       inputTensorNames, outputTensorNames);

  if (err != MODEL_NO_ERROR) {
    MLLM_ERROR("Failed to add node {} of type {} to graph {}: error code {}\n", nodeName, nodeType, graphName,
               static_cast<int>(err));
  }
}

bool QNNBackend::graphFinalize(const std::string& graphName) {
  auto it = qnnModelIndexMap_.find(graphName);
  if (it == qnnModelIndexMap_.end()) {
    MLLM_ERROR("Graph {} not found for finalization", graphName);
    return false;
  }

  int modelIndex = it->second;
  auto& qnnModel = qnnModels_[modelIndex];

  if (qnnModel->isGraphFinalized()) {
    MLLM_INFO("Graph {} is loaded from cache, skipping finalization", graphName);
    return true;
  }

  // Graph finalize
  if (MODEL_NO_ERROR != qnnModel->finalizeGraph(runtime_->profileHandle, nullptr)) {
    MLLM_ERROR("Failed to finalize graph: {}", graphName);
    return false;
  }

  qnnModel->freeCachedTensors();

  // Extract profiling info if enabled
  if (ProfilingLevel::OFF != profilingLevel_) { extractBackendProfilingInfo(runtime_->profileHandle); }

  return true;
}

void QNNBackend::graphExecute(const std::string& graphName, std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto it = qnnModelIndexMap_.find(graphName);
  if (it == qnnModelIndexMap_.end()) {
    MLLM_ERROR("Graph {} not found for execution", graphName);
    return;
  }
  auto model = qnnModels_[it->second];

  // Validate input size matches expected input count
  if (inputs.size() != model->getGraphInputTensorWrappers().size()) {
    MLLM_ERROR("Input size mismatch: expected {}, got {} for graph '{}'", 
               model->getGraphInputTensorWrappers().size(), inputs.size(), graphName);
    return;
  }

  // Prepare QNN input tensors by copying data from runtime inputs to graph input wrappers
  // This handles the case where input tensor sizes may differ between prefill and decode phases
  std::vector<Qnn_Tensor_t> qnn_inputs;
  std::vector<Qnn_Tensor_t> qnn_outputs;
  for (int i = 0; i < model->getGraphInputTensorWrappers().size(); i++) {
    auto wrapper = model->getGraphInputTensorWrappers()[i];
    auto& wrapper_tensor = wrapper->getDataContainer();
    const auto& runtime_input = inputs[i];

    // Validate input tensors
    if (runtime_input.isNil()) {
      MLLM_ERROR("Input tensor {} is nil for graph '{}'", i, graphName);
      return;
    }

    if (wrapper_tensor.isNil()) {
      MLLM_ERROR("Graph input wrapper {} for graph '{}' has no backing tensor", i, graphName);
      return;
    }

    // Check for size mismatches (can occur in decode phase where inputs may be smaller)
    size_t dst_bytes = wrapper_tensor.bytes();
    size_t src_bytes = runtime_input.bytes();
    if (dst_bytes != src_bytes) {
      MLLM_WARN("Graph '{}' input tensor {} byte-size mismatch: wrapper={} bytes, runtime input={} bytes. Copying "
                "min(dst, src), but this may truncate data.",
                graphName, i, dst_bytes, src_bytes);
    }

    if (dst_bytes > 0) {
      void* dst_ptr = wrapper_tensor.ptr<void>();
      if (!dst_ptr) {
        wrapper_tensor.alloc();
        dst_ptr = wrapper_tensor.ptr<void>();
      }

      const void* src_ptr = runtime_input.ptr<void>();
      size_t bytes_to_copy = std::min(dst_bytes, src_bytes);
      if (!src_ptr) {
        MLLM_ERROR("Runtime input tensor {} for graph '{}' has null data pointer", i, graphName);
        return;
      }
      if (dst_ptr && src_ptr && dst_ptr != src_ptr) {
        // Copy source data to destination buffer
        // This ensures that the graph input wrapper has the correct data for execution
        if (bytes_to_copy > 0) {
          std::memcpy(dst_ptr, src_ptr, bytes_to_copy);
        }
        
        // If source is smaller than destination, zero out the remaining bytes
        // This is important for decode phase where input tensors may be smaller than prefill
        // For example, decode phase may use [1, 1] input while wrapper expects [1, 128]
        // Note: In current implementation with full [1, 128] tensor, this should not trigger
        // but it's kept as a safety measure for future optimizations
        if (src_bytes < dst_bytes) {
          size_t remaining_bytes = dst_bytes - src_bytes;
          std::memset(static_cast<char*>(dst_ptr) + bytes_to_copy, 0, remaining_bytes);
          // Only log if zero-padding actually occurs (unexpected case)
          MLLM_WARN("[QNN graphExecute] Graph '{}' input tensor {}: zero-padded {} bytes (src={} bytes, dst={} bytes)", 
                    graphName, i, remaining_bytes, src_bytes, dst_bytes);
        }
      }
    }

    // Allocate and register the wrapper tensor with QNN allocator
    // QNNAllocator will handle registered memory descriptor when needed
    wrapper->alloc();
    qnn_inputs.push_back(*(wrapper->getNativeTensor()));
  }
  
  // Prepare QNN outputs in QNN order
  std::vector<Tensor> qnn_output_tensors;  // Temporary storage for QNN outputs
  for (int j = 0; j < model->getGraphOutputTensorWrappers().size(); j++) {
    // alloc and register qnn tensor
    model->getGraphOutputTensorWrappers()[j]->alloc();  // QNNAllocator will handle registered memory descriptor
    qnn_outputs.push_back(*(model->getGraphOutputTensorWrappers()[j]->getNativeTensor()));
    qnn_output_tensors.push_back(model->getGraphOutputTensorWrappers()[j]->getDataContainer());
  }

  CALL_QNN(runtime_->qnnInterface.graphExecute(model->getQnnGraph(), qnn_inputs.data(), qnn_inputs.size(), qnn_outputs.data(),
                                               qnn_outputs.size(), runtime_->profileHandle, nullptr));

  if (ProfilingLevel::OFF != profilingLevel_) { extractBackendProfilingInfo(runtime_->profileHandle); }

  // Debug: Print last output shape from QNN actual return order (before reordering)
  // Uncomment below for debugging output order issues
  // if (!qnn_output_tensors.empty()) {
  //   const auto& last_output = qnn_output_tensors.back();
  //   const auto& output_wrappers = model->getGraphOutputTensorWrappers();
  //   const auto& last_wrapper = output_wrappers.back();
  //   MLLM_INFO("[QNN Actual Return Order] Last output tensor '{}' shape: {}", 
  //             last_wrapper->getName(), last_output.shape());
  // }

  // Reorder outputs according to MLLM expected order
  const auto& expectedOrder = model->getExpectedOutputOrder();

  // Resize outputs to match QNN output count first
  outputs.resize(qnn_output_tensors.size());  // Ensure outputs has enough space for all QNN outputs
  if (!expectedOrder.empty() && expectedOrder.size() == qnn_output_tensors.size()) {
    // Debug: Log output order information
    // Uncomment below for debugging output order issues
    // MLLM_INFO("QNNBackend::graphExecute: Checking output order for graph '{}'", graphName);
    // MLLM_INFO("  MLLM Expected Output Order ({} outputs):", expectedOrder.size());
    // for (size_t i = 0; i < expectedOrder.size(); i++) {
    //   MLLM_INFO("    [{}] {}", i, expectedOrder[i]);
    // }
    // MLLM_INFO("  QNN Output Order ({} outputs):", model->getGraphOutputTensorWrappers().size());
    // for (size_t i = 0; i < model->getGraphOutputTensorWrappers().size(); i++) {
    //   auto wrapper = model->getGraphOutputTensorWrappers()[i];
    //   MLLM_INFO("    [{}] {}", i, wrapper->getName());
    // }

    // Check if reordering is needed
    // bool needs_reordering = false;
    // std::vector<std::pair<size_t, int>> mismatches;
    // for (size_t i = 0; i < expectedOrder.size(); i++) {
    //   const std::string& expected_name = expectedOrder[i];
    //   int qnn_index = model->getQnnOutputIndex(expected_name);
    //   if (qnn_index >= 0 && qnn_index < static_cast<int>(qnn_output_tensors.size())) {
    //     if (static_cast<int>(i) != qnn_index) {
    //       needs_reordering = true;
    //       mismatches.emplace_back(i, qnn_index);
    //     }
    //   }
    // }

    // Debug: Verification messages
    // Uncomment below for debugging output order issues
    // if (needs_reordering) {
    //   MLLM_INFO("  [VERIFICATION] QNN output order DIFFERS from MLLM expected order - REORDERING REQUIRED");
    //   for (const auto& [mllm_idx, qnn_idx] : mismatches) {
    //     MLLM_INFO("    Mismatch: MLLM[{}] expects '{}' but it's at QNN[{}]", 
    //               mllm_idx, expectedOrder[mllm_idx], qnn_idx);
    //   }
    // } else {
    //   MLLM_INFO("  [VERIFICATION] QNN output order MATCHES MLLM expected order - no reordering needed");
    // }

    // Reorder outputs according to expected order
    for (size_t i = 0; i < expectedOrder.size(); i++) {
      const std::string& expected_name = expectedOrder[i];
      int qnn_index = model->getQnnOutputIndex(expected_name);
      if (qnn_index >= 0 && qnn_index < static_cast<int>(qnn_output_tensors.size())) {
        outputs[i] = qnn_output_tensors[qnn_index];
        // Debug: Mapping information
        // Uncomment below for debugging output order issues
        // if (static_cast<int>(i) != qnn_index) {
        //   MLLM_INFO("  Mapping: MLLM[{}] = QNN[{}] (tensor: {}) [REORDERED]", i, qnn_index, expected_name);
        // } else {
        //   MLLM_INFO("  Mapping: MLLM[{}] = QNN[{}] (tensor: {}) [SAME]", i, qnn_index, expected_name);
        // }
      } else {
        MLLM_ERROR("QNNBackend::graphExecute: Failed to find QNN output index for tensor '{}' in graph '{}'", expected_name, graphName);
        // If mapping fails, we cannot safely reorder outputs
        // This is a critical error as we cannot determine the correct output order
        MLLM_ERROR("Cannot reorder outputs: missing QNN output index for tensor '{}'. Output order may be incorrect.", expected_name);
        // Note: We still try to copy what we can, but the order may be wrong
        if (i < qnn_output_tensors.size()) {
          outputs[i] = qnn_output_tensors[i];
        } else {
          MLLM_ERROR("Output index {} out of bounds (size: {})", i, qnn_output_tensors.size());
        }
      }
    }
  } else {
    // No expected order set or size mismatch, use QNN order as-is
    if (expectedOrder.empty()) {
      MLLM_WARN("QNNBackend::graphExecute: No expected output order set for graph '{}', using QNN order", graphName);
    } else {
      MLLM_WARN("QNNBackend::graphExecute: Expected output order size ({}) != outputs size ({}) for graph '{}', using QNN order",
                expectedOrder.size(), outputs.size(), graphName);
    }
    for (size_t i = 0; i < qnn_output_tensors.size(); i++) {
      outputs[i] = qnn_output_tensors[i];
    }
  }
}

bool QNNBackend::addTensor(const std::string& graphName, const std::string& tensorName, Qnn_TensorType_t type,
                           const Tensor& tensor, Qnn_QuantizeParams_t quantize) {
  auto it = qnnModelIndexMap_.find(graphName);
  if (it == qnnModelIndexMap_.end()) {
    MLLM_ERROR("Graph {} not found for adding tensor", graphName);
    return false;
  }

  int modelIndex = it->second;
  auto& qnnModel = qnnModels_[modelIndex];

  if (qnnModel->isGraphFinalized()) {
    MLLM_ERROR("Cannot add tensor {} to finalized graph {}", tensorName, graphName);
    return false;
  }

  ModelError_t err = qnnModel->addTensor(tensorName, type, tensor, quantize);
  if (err != MODEL_NO_ERROR) {
    MLLM_ERROR("Failed to add tensor {} to graph {}: error code {}", tensorName, graphName, static_cast<int>(err));
    return false;
  }

  return true;
}

bool QNNBackend::addStaticTensor(const std::string& graphName, const std::string& tensorName, const Tensor& tensor,
                                 Qnn_QuantizeParams_t quantize) {
  auto it = qnnModelIndexMap_.find(graphName);
  if (it == qnnModelIndexMap_.end()) {
    MLLM_ERROR("Graph {} not found for adding static tensor", graphName);
    return false;
  }

  int modelIndex = it->second;
  auto& qnnModel = qnnModels_[modelIndex];

  if (qnnModel->isGraphFinalized()) {
    MLLM_ERROR("Cannot add static tensor {} to finalized graph {}", tensorName, graphName);
    return false;
  }

  ModelError_t err = qnnModel->addStaticTensor(tensorName, tensor, quantize);
  if (err != MODEL_NO_ERROR) {
    MLLM_ERROR("Failed to add static tensor {} to graph {}: error code {}", tensorName, graphName, static_cast<int>(err));
    return false;
  }

  return true;
}

std::shared_ptr<QNNTensorWrapper> QNNBackend::getTensorWrapper(const std::string& graphName, const std::string& tensorName) {
  auto it = qnnModelIndexMap_.find(graphName);
  if (it == qnnModelIndexMap_.end()) {
    MLLM_ERROR("Graph {} not found for getting tensor wrapper", graphName);
    return nullptr;
  }

  int modelIndex = it->second;
  auto& qnnModel = qnnModels_[modelIndex];

  return qnnModel->getTensorWrapper(tensorName);
}

void QNNBackend::extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle) {
  // Extract profiling information from QNN backend
  // This is a placeholder implementation
  if (profileHandle == nullptr) { return; }

  const QnnProfile_EventId_t* profileEvents{nullptr};
  uint32_t numEvents{0};
  if (QNN_PROFILE_NO_ERROR != runtime_->qnnInterface.profileGetEvents(profileHandle, &profileEvents, &numEvents)) {
    MLLM_WARN("Failed to get profile events");
    return;
  }

  MLLM_INFO("Extracted {} profiling events", numEvents);
}

}  // namespace mllm::qnn
