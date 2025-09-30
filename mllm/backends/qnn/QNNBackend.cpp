#include "QNNBackend.hpp"
#include <cassert>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <memory>
#include "QNNUtils.hpp"
#include "QnnLog.h"
#include "mllm/backends/qnn/QNNAllocator.hpp"
#include "mllm/backends/qnn/op/QNNLinearOp.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::qnn {

QNNBackend::QNNBackend() : Backend(kQNN, createQNNAllocator()) {
  // register ops
  regOpFactory<QNNLinearOpFactory>();

  QnnLog_Level_t qnnLogLevel = QNN_LOG_LEVEL_INFO;  // default QNN log level
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

  MLLM_INFO("Created QNN graph: {}", graphName);
  return qnnModel;
}

void QNNBackend::graphAddNode(const std::string& graphName, const std::string& nodeName, const std::string& nodeType,
                              const std::vector<std::string>& inputTensorNames,
                              const std::vector<std::shared_ptr<QNNTensorWrapper>>& outputTensors,
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

  // Convert wrapper vectors to raw parameter vectors
  std::vector<std::shared_ptr<QNNParamTensorWrapper>> tensorParamsCopy = tensorParams;
  std::vector<std::shared_ptr<QNNParamScalarWrapper>> scalarParamsCopy = scalarParams;
  std::vector<std::string> inputNamesCopy = inputTensorNames;
  std::vector<std::shared_ptr<QNNTensorWrapper>> outputTensorsCopy = outputTensors;

  // Add node to the model
  ModelError_t err = qnnModel->addNode(QNN_OPCONFIG_VERSION_1, nodeName.c_str(), packageName.c_str(), nodeType.c_str(),
                                       tensorParamsCopy, scalarParamsCopy, inputNamesCopy, outputTensorsCopy);

  if (err != MODEL_NO_ERROR) {
    MLLM_ERROR("Failed to add node {} of type {} to graph {}: error code {}", nodeName, nodeType, graphName,
               static_cast<int>(err));
  } else {
    MLLM_INFO("Added node {} of type {} to graph {}", nodeName, nodeType, graphName);
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

  MLLM_INFO("Graph {} finalized successfully", graphName);
  return true;
}

void QNNBackend::graphExecute(const std::string& graphName) {
  auto model = qnnModels_[qnnModelIndexMap_[graphName]];

  std::vector<Qnn_Tensor_t> inputs;
  std::vector<Qnn_Tensor_t> outputs;
  for (int i = 0; i < model->getGraphInputTensorWrappers().size(); i++) {
    inputs.push_back(*(model->getGraphInputTensorWrappers()[i]->getNativeTensor()));
  }
  for (int j = 0; j < model->getGraphOutputTensorWrappers().size(); j++) {
    outputs.push_back(*(model->getGraphOutputTensorWrappers()[j]->getNativeTensor()));
  }

  CALL_QNN(runtime_->qnnInterface.graphExecute(model->getQnnGraph(), inputs.data(), inputs.size(), outputs.data(),
                                               outputs.size(), runtime_->profileHandle, nullptr));
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