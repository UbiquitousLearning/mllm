#include <inttypes.h>

#include <cstring>
#include <fstream>
#include <iostream>

#include "QNNBackend.hpp"
#include "BuildId.hpp"
#include "DataUtil.hpp"
#include "Logger.hpp"
#include "PAL/Directory.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"
#include "PAL/StringOp.hpp"
#include "PAL/DynamicLoading.hpp"
#include "PAL/GetOpt.hpp"
#include "QnnSampleAppUtils.hpp"
#include "QnnTypes.h"
#include "QnnWrapperUtils.hpp"
#include "DynamicLoadUtil.hpp"
#include "Types.hpp"
#include "op/QNNAdd.hpp"
#include "op/QNNLinear.hpp"
#include "op/QNNMatmul.hpp"
#include "op/QNNMul.hpp"
#include "op/QNNScale.hpp"
#include "op/QNNSiLU.hpp"
#include "op/QNNSoftMax.hpp"
#include "op/QNNView.hpp"

#include "QnnTypeMacros.hpp"
#include "IOTensor.hpp"

using namespace qnn;
using namespace qnn::tools;
using namespace qnn::tools::sample_app;

// Flag to determine if Backend should node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

namespace mllm {

const std::string QNNBackend::s_defaultOutputPath = "./output";

void QNNBackend::registerOps() {
    addCreator(ADD, (QNNBackend::Creator *)new QNNAddCreator());
    // addCreator(CAUSALMASK, (QNNBackend::Creator *)(new QNNCausalMaskCreator()));
    addCreator(MATMUL, (QNNBackend::Creator *)(new QNNMatmulCreator()));
    // addCreator(RMSNORM, (QNNBackend::Creator *)(new QNNRMSNormCreator()));
    // addCreator(ROPE, (QNNBackend::Creator *)(new QNNRoPECreator()));
    addCreator(SCALE, (QNNBackend::Creator *)(new QNNScaleCreator()));
    addCreator(SILU, (QNNBackend::Creator *)(new QNNSiLUCreator()));
    addCreator(SOFTMAX, (QNNBackend::Creator *)(new QNNSoftMaxCreator()));
    addCreator(LINEAR, (QNNBackend::Creator *)(new QNNLinearCreator()));
    // addCreator(ATTENTION, (QNNBackend::Creator *)(new QNNAttentionCreator()));
    // addCreator(EMBEDDING, (QNNBackend::Creator *)(new QNNEmbeddingCreator()));
    addCreator(MUL, (QNNBackend::Creator *)(new QNNMulCreator()));
    addCreator(VIEW, (QNNBackend::Creator *)(new QNNViewCreator()));
    // addCreator(KVCACHE, (QNNBackend::Creator *)(new QNNKVCacheCreator()));
}

QNNBackend::QNNBackend(shared_ptr<MemoryManager> mm) : Backend(mm) {
  
    if (!log::initializeLogging()) {
      std::cerr << "ERROR: Unable to initialize logging!\n";
      return;
    }
    // TODO: make debug level configuable
    log::setLogLevel(QnnLog_Level_t::QNN_LOG_LEVEL_ERROR);

#ifdef QNN_ZH
    // std::string modelPath = "/qnn-projects/QNN-test-libs/example_libs/x86_64-linux-clang/libqnn_model_float.so";
    // std::string backEndPath = "/qnn-projects/QNN-test-libs/libQnnCpu.so";
    std::string backEndPath = "/qnn-projects/QNN-test-libs/libQnnHtp.so";
    std::string inputListPaths = "/qnn-projects/mllm/bin/input-list.txt";
    // std::string opPackagePaths = "/qnn-projects/QNN-test-libs/libQnnCpuOpPackageExample.so:QnnOpPackage_interfaceProvider";
    std::string opPackagePaths = "/qnn-projects/QNN-test-libs/llama-op-package/libQnnLLaMAPackage.so:LLaMAPackageInterfaceProvider";
#elifdef QNN_ARM
    std::string backEndPath = "libQnnHtp.so";
    std::string inputListPaths = "./qnn/input-list.txt";
    std::string opPackagePaths = "./qnn/llama-op-hexagon75/libQnnLLaMAPackage.so:LLaMAPackageInterfaceProvider";
#else
    std::string modelPath = "/mllm/qualcomm_ai_engine_direct_new/examples/QNN/example_libs/x86_64-linux-clang/libqnn_model_float.so";
    std::string backEndPath = "/mllm/qualcomm_ai_engine_direct_new/lib/x86_64-linux-clang/libQnnHtp.so";
    std::string inputListPaths = "/mllm/test_zh/input_list_float.txt";
    std::string opPackagePaths = "/mllm/LLaMAOpPackageHtp/LLaMAPackage/build/x86_64-linux-clang/libQnnLLaMAPackage.so:LLaMAPackageInterfaceProvider";
#endif

    // TODO: make these configuable
    m_debug = true;
    m_outputDataType = iotensor::OutputDataType::FLOAT_ONLY;
    m_inputDataType = iotensor::InputDataType::FLOAT;
    m_profilingLevel = ProfilingLevel::OFF;
    m_dumpOutputs = true;
    m_isBackendInitialized = false;
    m_isContextCreated = false;

    // config path strings
    split(m_inputListPaths, inputListPaths, ',');
    split(m_opPackagePaths, opPackagePaths, ',');
    if (m_outputPath.empty()) {
        m_outputPath = s_defaultOutputPath;
    }

    if (backEndPath.empty()) {
      std::exit(EXIT_FAILURE);
    }

    // if (inputListPaths.empty()) {
    //   std::exit(EXIT_FAILURE);
    // }

    QNN_INFO("Backend: %s", backEndPath.c_str());

    // QnnFunctionPointers qnnFunctionPointers;
    // Load backend and model .so and validate all the required function symbols are resolved
    auto statusCode = dynamicloadutil::getQnnFunctionPointers(backEndPath,
                                                              "",
                                                              &m_qnnFunctionPointers,
                                                              &m_backendLibraryHandle,
                                                              false,
                                                              &m_modelHandle);

    if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
      if (dynamicloadutil::StatusCode::FAIL_LOAD_BACKEND == statusCode) {
        exitWithMessage(
            "Error initializing QNN Function Pointers: could not load backend: " + backEndPath,
            EXIT_FAILURE);
      } else if (dynamicloadutil::StatusCode::FAIL_LOAD_MODEL == statusCode) {
        exitWithMessage(
            "Error initializing QNN Function Pointers: could not load model: ",
            EXIT_FAILURE);
      } else {
        exitWithMessage("Error initializing QNN Function Pointers", EXIT_FAILURE);
      }
    }

    // cause we build graph in runtime, the freeGraphInfoFnHandle should be assigned here
    m_qnnFunctionPointers.freeGraphInfoFnHandle = this->QnnModel_freeGraphsInfo;

    // init qnn resources and create a graph
    this->graphInitialize();
    this->registerOps();
}

void QNNBackend::release() {
    if (StatusCode::SUCCESS != this->freeContext()) {
        this->reportError("Context Free failure");
    }

    auto devicePropertySupportStatus = this->isDevicePropertySupported();
    if (StatusCode::FAILURE != devicePropertySupportStatus) {
        auto freeDeviceStatus = this->freeDevice();
        if (StatusCode::SUCCESS != freeDeviceStatus) {
            this->reportError("Device Free failure");
        }
    }
}

void QNNBackend::onSetUpStart(vector<shared_ptr<Tensor>> &inputs) {
  #ifdef DEBUG
    std::cout << "onSetUpStart" << std::endl;
  #endif
    // add input tensor to qnn
    uint32_t dimensionsInput[4];
    for (int i = 0; i < 4; i++) {
        dimensionsInput[i] = inputs[0]->shape()[i];
    }
    this->modelAddTensor(inputs[0]->name().c_str(), (Qnn_Tensor_t){
                                                        .version = QNN_TENSOR_VERSION_1,
                                                        {.v1 = {
                                                             .id = 0,
                                                             .name = inputs[0]->name().c_str(),
                                                             .type = QNN_TENSOR_TYPE_APP_WRITE,
                                                             .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                             .dataType = QNN_DATATYPE_UFIXED_POINT_8,
                                                             .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                                QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                                {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                             .rank = 4,
                                                             .dimensions = dimensionsInput,
                                                             .memType = QNN_TENSORMEMTYPE_RAW,
                                                             {.clientBuf = {.data = nullptr,
                                                                            .dataSize = 0}}}}});
}

void QNNBackend::onExecuteStart(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    graphFinilize();
    for(auto &input : inputs) {
        std::cout << "input dtype:" << input->dtype() << std::endl;
        // input->printData<float>();
        inputBuffers.push_back(input->hostPtr<uint8_t>());
    }
    inputBufferMap.insert(std::make_pair("graph", inputBuffers));
    for(auto &output : outputs) {
        std::cout << "output dtype:" << output->dtype() << std::endl;
        output->alloc();
        // output->printData<float>();
        outputBuffers.push_back(output->hostPtr<uint8_t>());
    }
    outputBufferMap.insert(std::make_pair("graph", outputBuffers));
}

void QNNBackend::onExecuteEnd() {
    graphExecute(inputBufferMap, outputBufferMap);
}

std::string QNNBackend::getBackendBuildId() {
  char* backendBuildId{nullptr};
  if (QNN_SUCCESS !=
      m_qnnFunctionPointers.qnnInterface.backendGetBuildId((const char**)&backendBuildId)) {
    QNN_ERROR("Unable to get build Id from the backend.");
  }
  return (backendBuildId == nullptr ? std::string("") : std::string(backendBuildId));
}

int32_t QNNBackend::graphInitialize() {
    QNN_INFO("qnn-backend    build version: %s", getBuildId().c_str());
    QNN_INFO("Backend        build version: %s", getBackendBuildId().c_str());

    if (StatusCode::SUCCESS != this->initialize()) {
        this->reportError("Initialization failure");
    }

    if (StatusCode::SUCCESS != this->initializeBackend()) {
        this->reportError("Backend Initialization failure");
    }

    auto devicePropertySupportStatus = this->isDevicePropertySupported();
    if (StatusCode::FAILURE != devicePropertySupportStatus) {
        auto createDeviceStatus = this->createDevice();
        if (StatusCode::SUCCESS != createDeviceStatus) {
            this->reportError("Device Creation failure");
        }
    }

    if (StatusCode::SUCCESS != this->initializeProfiling()) {
        this->reportError("Profiling Initialization failure");
    }

    if (StatusCode::SUCCESS != this->registerOpPackages()) {
        this->reportError("Register Op Packages failure");
    }

    if (StatusCode::SUCCESS != this->createContext()) {
        this->reportError("Context Creation failure");
    }
    
    // initialize graph info, set graph info, graph count
    // acting the same as composeGraphs
    const QnnGraph_Config_t **graphConfigs = nullptr;
    qnn_wrapper_api::ModelError_t err = qnn_wrapper_api::MODEL_NO_ERROR;
    VALIDATE(qnn_wrapper_api::getQnnGraphConfigFromInfo(
                 "mllmQnnModel", (const qnn_wrapper_api::GraphConfigInfo_t **)m_graphConfigsInfo, m_graphConfigsInfoCount, graphConfigs),
             err);
    VALIDATE(qnnModel.initialize(m_backendHandle,
                                      m_qnnFunctionPointers.qnnInterface,
                                      m_context,
                                      "mllmQnnModel",
                                      m_debug,
                                      DO_GRAPH_NODE_VALIDATIONS,
                                      graphConfigs),
             err);
    return 0;
}

qnn_wrapper_api::ModelError_t QNNBackend::graphAddNode(string name,
                                                       string nodeType,
                                                       std::vector<const char *> inputTensorNames,
                                                       std::vector<Qnn_Tensor_t> outputTensors,
                                                       std::vector<Qnn_Param_t> params,
                                                       string packageName) {
    qnn_wrapper_api::ModelError_t err = qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR;
    Qnn_Param_t* paramsPtr = nullptr;
    if (params.size() > 0) {
        paramsPtr = params.data();
    }
    VALIDATE(qnnModel.addNode(
                 QNN_OPCONFIG_VERSION_1,  // Op_Config_t Version
                 name.c_str(),            // Node Name
                 packageName.c_str(),     // Package Name
                 nodeType.c_str(),        // Qnn Node Type
                 paramsPtr,                 // Node Params
                 params.size(),                       // Num Node Params
                 inputTensorNames.data(), // Input Tensor Names
                 inputTensorNames.size(), // Num Input Tensor Names
                 outputTensors.data(),    // Output Tensors
                 outputTensors.size()     // Num Output Tensors
                 ),
             err);
    return err;
}

qnn_wrapper_api::ModelError_t QNNBackend::graphFinilize() {
    // Add all models to array to get graphsInfo
    qnn_wrapper_api::QnnModel *models[] = {&qnnModel};
    m_graphsCount = 1;
    // Populate the constructed graphs in provided output variables
    qnn_wrapper_api::ModelError_t err = qnn_wrapper_api::MODEL_NO_ERROR;
    VALIDATE(getGraphInfoFromModels(*models, m_graphsCount, &m_graphsInfo), err);
    // Graph finalize
    if (QNN_GRAPH_NO_ERROR != m_qnnFunctionPointers.qnnInterface.graphFinalize((*m_graphsInfo)[0].graph, m_profileBackendHandle, nullptr)) {
        return qnn_wrapper_api::ModelError_t::MODEL_GRAPH_ERROR;
    }
    if (ProfilingLevel::OFF != m_profilingLevel) {
        extractBackendProfilingInfo(m_profileBackendHandle);
    }

    return qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR;
}

qnn_wrapper_api::ModelError_t QNNBackend::modelAddTensor(const char *nodeName, Qnn_Tensor_t tensor) {
    return qnnModel.addTensor(nodeName, tensor);
}

ErrorCode QNNBackend::graphExecute() {
    // currently only call executeGraphs
    auto result = this->executeGraphs();
    if(result != StatusCode::SUCCESS) {
        return ErrorCode::INVALID_VALUE;
    }
    return NO_ERROR;
}

ErrorCode QNNBackend::graphExecute(std::map< std::string, std::vector<uint8_t*>> inputBufferMap, std::map< std::string, std::vector<uint8_t*>> outputBufferMap) {
    // currently only call executeGraphs
    auto result = this->executeGraphs(inputBufferMap, outputBufferMap);
    if(result != StatusCode::SUCCESS) {
        return ErrorCode::INVALID_VALUE;
    }
    return NO_ERROR;
}


// Initialize QnnSampleApp. Things it does:
//  1. Create output directory
//  2. Read all input list paths provided
//      during creation.
StatusCode QNNBackend::initialize() {
  // Create Output Directory
  if (m_dumpOutputs && !::pal::FileOp::checkFileExists(m_outputPath) &&
      !pal::Directory::makePath(m_outputPath)) {
    exitWithMessage("Could not create output directory: " + m_outputPath, EXIT_FAILURE);
  }
  // Read Input File List
  bool readSuccess;
  std::tie(m_inputFileLists, readSuccess) = readInputLists(m_inputListPaths);
  if (!readSuccess) {
    exitWithMessage("Could not read input lists", EXIT_FAILURE);
  }

  // initialize logging in the backend
  if (log::isLogInitialized()) {
    auto logCallback = log::getLogCallback();
    auto logLevel    = log::getLogLevel();
    QNN_INFO("Initializing logging in the backend. Callback: [%p], Log Level: [%d]",
             logCallback,
             logLevel);
    if (QNN_SUCCESS !=
        m_qnnFunctionPointers.qnnInterface.logCreate(logCallback, logLevel, &m_logHandle)) {
      QNN_WARN("Unable to initialize logging in the backend.");
    }
  } else {
    QNN_WARN("Logging not available in the backend.");
  }
  return StatusCode::SUCCESS;
}

StatusCode QNNBackend::initializeProfiling() {
  if (ProfilingLevel::OFF != m_profilingLevel) {
    QNN_INFO("Profiling turned on; level = %d", m_profilingLevel);
    if (ProfilingLevel::BASIC == m_profilingLevel) {
      QNN_INFO("Basic profiling requested. Creating Qnn Profile object.");
      if (QNN_PROFILE_NO_ERROR !=
          m_qnnFunctionPointers.qnnInterface.profileCreate(
              m_backendHandle, QNN_PROFILE_LEVEL_BASIC, &m_profileBackendHandle)) {
        QNN_WARN("Unable to create profile handle in the backend.");
        return StatusCode::FAILURE;
      }
    } else if (ProfilingLevel::DETAILED == m_profilingLevel) {
      QNN_INFO("Detailed profiling requested. Creating Qnn Profile object.");
      if (QNN_PROFILE_NO_ERROR !=
          m_qnnFunctionPointers.qnnInterface.profileCreate(
              m_backendHandle, QNN_PROFILE_LEVEL_DETAILED, &m_profileBackendHandle)) {
        QNN_ERROR("Unable to create profile handle in the backend.");
        return StatusCode::FAILURE;
      }
    }
  }
  return StatusCode::SUCCESS;
}

// Simple method to report error from app to lib.
void QNNBackend::reportError(const std::string& err) {
  QNN_ERROR("%s", err.c_str());
  exit(1);
}

// Initialize a QnnBackend.
StatusCode QNNBackend::initializeBackend() {
  auto qnnStatus = m_qnnFunctionPointers.qnnInterface.backendCreate(
      m_logHandle, (const QnnBackend_Config_t**)m_backendConfig, &m_backendHandle);
  if (QNN_BACKEND_NO_ERROR != qnnStatus) {
    QNN_ERROR("Could not initialize backend due to error = %d", qnnStatus);
    return StatusCode::FAILURE;
  }
  QNN_INFO("Initialize Backend Returned Status = %d", qnnStatus);
  m_isBackendInitialized = true;
  return StatusCode::SUCCESS;
}

// Terminate the backend after done.
StatusCode QNNBackend::terminateBackend() {
  if ((m_isBackendInitialized && nullptr != m_qnnFunctionPointers.qnnInterface.backendFree) &&
      QNN_BACKEND_NO_ERROR != m_qnnFunctionPointers.qnnInterface.backendFree(m_backendHandle)) {
    QNN_ERROR("Could not terminate backend");
    return StatusCode::FAILURE;
  }
  m_isBackendInitialized = false;
  return StatusCode::SUCCESS;
}

// Register op packages and interface providers supplied during
// object creation. If there are multiple op packages, register
// them sequentially in the order provided.
StatusCode QNNBackend::registerOpPackages() {
  const size_t pathIdx              = 0;
  const size_t interfaceProviderIdx = 1;
  for (auto const& opPackagePath : m_opPackagePaths) {
    std::vector<std::string> opPackage;
    split(opPackage, opPackagePath, ':');
    QNN_DEBUG("opPackagePath: %s", opPackagePath.c_str());
    const char* target     = nullptr;
    const size_t targetIdx = 2;
    if (opPackage.size() != 2 && opPackage.size() != 3) {
      QNN_ERROR("Malformed opPackageString provided: %s", opPackagePath.c_str());
      return StatusCode::FAILURE;
    }
    if (opPackage.size() == 3) {
      target = (char*)opPackage[targetIdx].c_str();
    }
    if (nullptr == m_qnnFunctionPointers.qnnInterface.backendRegisterOpPackage) {
      QNN_ERROR("backendRegisterOpPackageFnHandle is nullptr.");
      return StatusCode::FAILURE;
    }
    if (QNN_BACKEND_NO_ERROR != m_qnnFunctionPointers.qnnInterface.backendRegisterOpPackage(
                                    m_backendHandle,
                                    (char*)opPackage[pathIdx].c_str(),
                                    (char*)opPackage[interfaceProviderIdx].c_str(),
                                    target)) {
      QNN_ERROR("Could not register Op Package: %s and interface provider: %s",
                opPackage[pathIdx].c_str(),
                opPackage[interfaceProviderIdx].c_str());
      return StatusCode::FAILURE;
    }
    QNN_INFO("Registered Op Package: %s and interface provider: %s",
             opPackage[pathIdx].c_str(),
             opPackage[interfaceProviderIdx].c_str());
  }
  return StatusCode::SUCCESS;
}

// Create a Context in a backend.
StatusCode QNNBackend::createContext() {
  if (QNN_CONTEXT_NO_ERROR != m_qnnFunctionPointers.qnnInterface.contextCreate(
                                  m_backendHandle,
                                  m_deviceHandle,
                                  (const QnnContext_Config_t**)&m_contextConfig,
                                  &m_context)) {
    QNN_ERROR("Could not create context");
    return StatusCode::FAILURE;
  }
  m_isContextCreated = true;
  return StatusCode::SUCCESS;
}

// Free context after done.
StatusCode QNNBackend::freeContext() {
  if (QNN_CONTEXT_NO_ERROR !=
      m_qnnFunctionPointers.qnnInterface.contextFree(m_context, m_profileBackendHandle)) {
    QNN_ERROR("Could not free context");
    return StatusCode::FAILURE;
  }
  m_isContextCreated = false;
  return StatusCode::SUCCESS;
}

// Calls composeGraph function in QNN's model.so.
// composeGraphs is supposed to populate graph related
// information in m_graphsInfo and m_graphsCount.
// m_debug is the option supplied to composeGraphs to
// say that all intermediate tensors including output tensors
// are expected to be read by the app.
StatusCode QNNBackend::composeGraphs() {
  auto returnStatus = StatusCode::SUCCESS;
  if (qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR !=
      m_qnnFunctionPointers.composeGraphsFnHandle(
          m_backendHandle,
          m_qnnFunctionPointers.qnnInterface,
          m_context,
          (const qnn_wrapper_api::GraphConfigInfo_t**)m_graphConfigsInfo,
          m_graphConfigsInfoCount,
          &m_graphsInfo,
          &m_graphsCount,
          m_debug,
          log::getLogCallback(),
          log::getLogLevel())) {
    QNN_ERROR("Failed in composeGraphs()");
    returnStatus = StatusCode::FAILURE;
  }
  return returnStatus;
}

StatusCode QNNBackend::finalizeGraphs() {
  for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    if (QNN_GRAPH_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.graphFinalize(
            (*m_graphsInfo)[graphIdx].graph, m_profileBackendHandle, nullptr)) {
      return StatusCode::FAILURE;
    }
  }
  if (ProfilingLevel::OFF != m_profilingLevel) {
    extractBackendProfilingInfo(m_profileBackendHandle);
  }

  return StatusCode::SUCCESS;
}

StatusCode QNNBackend::extractBackendProfilingInfo(
    Qnn_ProfileHandle_t profileHandle) {
  if (nullptr == m_profileBackendHandle) {
    QNN_ERROR("Backend Profile handle is nullptr; may not be initialized.");
    return StatusCode::FAILURE;
  }
  const QnnProfile_EventId_t* profileEvents{nullptr};
  uint32_t numEvents{0};
  if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileGetEvents(
                                  profileHandle, &profileEvents, &numEvents)) {
    QNN_ERROR("Failure in profile get events.");
    return StatusCode::FAILURE;
  }
  QNN_DEBUG("ProfileEvents: [%p], numEvents: [%d]", profileEvents, numEvents);
  for (size_t event = 0; event < numEvents; event++) {
    extractProfilingEvent(*(profileEvents + event));
    extractProfilingSubEvents(*(profileEvents + event));
  }
  return StatusCode::SUCCESS;
}

StatusCode QNNBackend::extractProfilingSubEvents(
    QnnProfile_EventId_t profileEventId) {
  const QnnProfile_EventId_t* profileSubEvents{nullptr};
  uint32_t numSubEvents{0};
  if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileGetSubEvents(
                                  profileEventId, &profileSubEvents, &numSubEvents)) {
    QNN_ERROR("Failure in profile get sub events.");
    return StatusCode::FAILURE;
  }
  QNN_DEBUG("ProfileSubEvents: [%p], numSubEvents: [%d]", profileSubEvents, numSubEvents);
  for (size_t subEvent = 0; subEvent < numSubEvents; subEvent++) {
    extractProfilingEvent(*(profileSubEvents + subEvent));
    extractProfilingSubEvents(*(profileSubEvents + subEvent));
  }
  return StatusCode::SUCCESS;
}

StatusCode QNNBackend::extractProfilingEvent(
    QnnProfile_EventId_t profileEventId) {
  QnnProfile_EventData_t eventData;
  if (QNN_PROFILE_NO_ERROR !=
      m_qnnFunctionPointers.qnnInterface.profileGetEventData(profileEventId, &eventData)) {
    QNN_ERROR("Failure in profile get event type.");
    return StatusCode::FAILURE;
  }
  QNN_DEBUG("Printing Event Info - Event Type: [%d], Event Value: [%" PRIu64
            "], Event Identifier: [%s], Event Unit: [%d]",
            eventData.type,
            eventData.value,
            eventData.identifier,
            eventData.unit);
  return StatusCode::SUCCESS;
}

StatusCode QNNBackend::verifyFailReturnStatus(Qnn_ErrorHandle_t errCode) {
  auto returnStatus = StatusCode::FAILURE;
  switch (errCode) {
    case QNN_COMMON_ERROR_SYSTEM_COMMUNICATION:
      returnStatus = StatusCode::FAILURE_SYSTEM_COMMUNICATION_ERROR;
      break;
    case QNN_COMMON_ERROR_SYSTEM:
      returnStatus = StatusCode::FAILURE_SYSTEM_ERROR;
      break;
    case QNN_COMMON_ERROR_NOT_SUPPORTED:
      returnStatus = StatusCode::QNN_FEATURE_UNSUPPORTED;
      break;
    default:
      break;
  }
  return returnStatus;
}

StatusCode QNNBackend::isDevicePropertySupported() {
  if (nullptr != m_qnnFunctionPointers.qnnInterface.propertyHasCapability) {
    auto qnnStatus =
        m_qnnFunctionPointers.qnnInterface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
    if (QNN_PROPERTY_NOT_SUPPORTED == qnnStatus) {
      QNN_WARN("Device property is not supported");
    }
    if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnStatus) {
      QNN_ERROR("Device property is not known to backend");
      return StatusCode::FAILURE;
    }
  }
  return StatusCode::SUCCESS;
}

StatusCode QNNBackend::createDevice() {
  if (nullptr != m_qnnFunctionPointers.qnnInterface.deviceCreate) {
    auto qnnStatus =
        m_qnnFunctionPointers.qnnInterface.deviceCreate(m_logHandle, nullptr, &m_deviceHandle);
    if (QNN_SUCCESS != qnnStatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnStatus) {
      QNN_ERROR("Failed to create device");
      return verifyFailReturnStatus(qnnStatus);
    }
  }
  return StatusCode::SUCCESS;
}

StatusCode QNNBackend::freeDevice() {
  if (nullptr != m_qnnFunctionPointers.qnnInterface.deviceFree) {
    auto qnnStatus = m_qnnFunctionPointers.qnnInterface.deviceFree(m_deviceHandle);
    if (QNN_SUCCESS != qnnStatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnStatus) {
      QNN_ERROR("Failed to free device");
      return verifyFailReturnStatus(qnnStatus);
    }
  }
  return StatusCode::SUCCESS;
}

// executeGraphs() that is currently used by qnn-sample-app's main.cpp.
// This function runs all the graphs present in model.so by reading
// inputs from input_list based files and writes output to .raw files.
StatusCode QNNBackend::executeGraphs() {
  auto returnStatus = StatusCode::SUCCESS;
  for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    QNN_DEBUG("Starting execution for graphIdx: %d", graphIdx);
    if (graphIdx >= m_inputFileLists.size()) {
      QNN_ERROR("No Inputs available for: %d", graphIdx);
      returnStatus = StatusCode::FAILURE;
      break;
    }
    Qnn_Tensor_t* inputs  = nullptr;
    Qnn_Tensor_t* outputs = nullptr;
    if (iotensor::StatusCode::SUCCESS !=
        m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs, (*m_graphsInfo)[graphIdx])) {
      QNN_ERROR("Error in setting up Input and output Tensors for graphIdx: %d", graphIdx);
      returnStatus = StatusCode::FAILURE;
      break;
    }
    auto inputFileList = m_inputFileLists[graphIdx];
    auto graphInfo     = (*m_graphsInfo)[graphIdx];
    if (!inputFileList.empty()) {
      size_t totalCount = inputFileList[0].size();
      while (!inputFileList[0].empty()) {
        size_t startIdx = (totalCount - inputFileList[0].size());
        if (iotensor::StatusCode::SUCCESS !=
            m_ioTensor.populateInputTensors(
                graphIdx, inputFileList, inputs, graphInfo, m_inputDataType)) {
          returnStatus = StatusCode::FAILURE;
        }
        if (StatusCode::SUCCESS == returnStatus) {
          QNN_DEBUG("Successfully populated input tensors for graphIdx: %d", graphIdx);
          Qnn_ErrorHandle_t executeStatus = QNN_GRAPH_NO_ERROR;
          executeStatus =
              m_qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                              inputs,
                                                              graphInfo.numInputTensors,
                                                              outputs,
                                                              graphInfo.numOutputTensors,
                                                              m_profileBackendHandle,
                                                              nullptr);
          if (QNN_GRAPH_NO_ERROR != executeStatus) {
            returnStatus = StatusCode::FAILURE;
          }
          if (StatusCode::SUCCESS == returnStatus) {
            QNN_DEBUG("Successfully executed graphIdx: %d ", graphIdx);
            if (iotensor::StatusCode::SUCCESS !=
                m_ioTensor.writeOutputTensors(graphIdx,
                                              startIdx,
                                              graphInfo.graphName,
                                              outputs,
                                              graphInfo.numOutputTensors,
                                              m_outputDataType,
                                              m_graphsCount,
                                              m_outputPath)) {
              returnStatus = StatusCode::FAILURE;
            }
          }
        }
        if (StatusCode::SUCCESS != returnStatus) {
          QNN_ERROR("Execution of Graph: %d failed!", graphIdx);
          break;
        }
      }
    }
    m_ioTensor.tearDownInputAndOutputTensors(
        inputs, outputs, graphInfo.numInputTensors, graphInfo.numOutputTensors);
    inputs  = nullptr;
    outputs = nullptr;
    if (StatusCode::SUCCESS != returnStatus) {
      break;
    }
  }

  qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
  m_graphsInfo = nullptr;
  return returnStatus;
}


// executeGraphs() that load input/output buffers from CPU context
// inputBufferMap and outputBufferMap: graph_name -> graph input/output CPU buffers.
StatusCode QNNBackend::executeGraphs(std::map< std::string, std::vector<uint8_t*>> inputBufferMap, std::map< std::string, std::vector<uint8_t*>> outputBufferMap) {
  auto returnStatus = StatusCode::SUCCESS;
  for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
    QNN_DEBUG("Starting execution for graphIdx: %d", graphIdx);
    if (graphIdx >= inputBufferMap.size()) {
        QNN_ERROR("No Inputs available for: %d", graphIdx);
        returnStatus = StatusCode::FAILURE;
        break;
    }
    Qnn_Tensor_t* inputs  = nullptr;
    Qnn_Tensor_t* outputs = nullptr;
    if (iotensor::StatusCode::SUCCESS !=
        m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs, (*m_graphsInfo)[graphIdx])) {
      QNN_ERROR("Error in setting up Input and output Tensors for graphIdx: %d", graphIdx);
      returnStatus = StatusCode::FAILURE;
      break;
    }
    // auto inputFileList = m_inputFileLists[graphIdx];
    auto graphInfo     = (*m_graphsInfo)[graphIdx];
    if (!inputBufferMap.empty()) {

      // Todo only one graph now. Name "graph"
      size_t totalCount = inputBufferMap["graph"].size();
      if (iotensor::StatusCode::SUCCESS !=
            m_ioTensor.populateInputTensors(
                graphIdx, inputBufferMap["graph"], inputs, graphInfo, m_inputDataType)) {
          returnStatus = StatusCode::FAILURE;
      }

      size_t startIdx = 0;
      
      if (StatusCode::SUCCESS == returnStatus) {
          QNN_DEBUG("Successfully populated input tensors for graphIdx: %d", graphIdx);
          Qnn_ErrorHandle_t executeStatus = QNN_GRAPH_NO_ERROR;
          executeStatus =
              m_qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                              inputs,
                                                              graphInfo.numInputTensors,
                                                              outputs,
                                                              graphInfo.numOutputTensors,
                                                              m_profileBackendHandle,
                                                              nullptr);
          if (QNN_GRAPH_NO_ERROR != executeStatus) {
            returnStatus = StatusCode::FAILURE;
          }
          if (StatusCode::SUCCESS == returnStatus) {
            QNN_DEBUG("Successfully executed graphIdx: %d ", graphIdx);
            for (int oi=0; oi < graphInfo.numOutputTensors; oi ++) {
                auto output = outputs[oi];

                // m_ioTensor.writeOutputTensor(&output, outputBufferMap["graph"][oi]);
                memcpy(outputBufferMap["graph"][oi], output.v1.clientBuf.data, output.v1.clientBuf.dataSize);
            }
            
          }
        }
        if (StatusCode::SUCCESS != returnStatus) {
          QNN_ERROR("Execution of Graph: %d failed!", graphIdx);
          break;
        }
      }

      m_ioTensor.tearDownInputAndOutputTensors(
        inputs, outputs, graphInfo.numInputTensors, graphInfo.numOutputTensors);
      inputs  = nullptr;
      outputs = nullptr;
      if (StatusCode::SUCCESS != returnStatus) {
        break;
      }



    }

  qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
  m_graphsInfo = nullptr;
  return returnStatus;
}



}