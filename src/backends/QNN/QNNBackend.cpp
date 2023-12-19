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
#include "QnnWrapperUtils.hpp"
#include "DynamicLoadUtil.hpp"
#include "Types.hpp"
#include "op/QNNAdd.hpp"

using namespace qnn;
using namespace qnn::tools;
using namespace qnn::tools::sample_app;

// Flag to determine if Backend should node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

namespace mllm {

const std::string QNNBackend::s_defaultOutputPath = "./output";

void QNNBackend::QnnBackendInitialize(QnnFunctionPointers qnnFunctionPointers,
                                       std::string inputListPaths,
                                       std::string opPackagePaths,
                                       void* backendLibraryHandle,
                                       std::string outputPath,
                                       bool debug,
                                       iotensor::OutputDataType outputDataType,
                                       iotensor::InputDataType inputDataType,
                                       sample_app::ProfilingLevel profilingLevel,
                                       bool dumpOutputs,
                                       std::string cachedBinaryPath,
                                       std::string saveBinaryName)
    {

      m_qnnFunctionPointers = qnnFunctionPointers;
      m_outputPath = outputPath;
      m_saveBinaryName = saveBinaryName;
      m_cachedBinaryPath = cachedBinaryPath;
      m_debug = debug;
      m_outputDataType = outputDataType;
      m_inputDataType = inputDataType;
      m_profilingLevel = profilingLevel;
      m_dumpOutputs = dumpOutputs;
      m_backendLibraryHandle = backendLibraryHandle;
      m_isBackendInitialized = false;
      m_isContextCreated = false; 
      
      split(m_inputListPaths, inputListPaths, ',');
      split(m_opPackagePaths, opPackagePaths, ',');
      if (m_outputPath.empty()) {
        m_outputPath = s_defaultOutputPath;
      }

  return;
}

QNNBackend::QNNBackend(shared_ptr<MemoryManager> mm) : Backend(mm) {
  
    if (!log::initializeLogging()) {
      std::cerr << "ERROR: Unable to initialize logging!\n";
      return ;
    }

    enum OPTIONS {
      OPT_HELP             = 0,
      OPT_MODEL            = 1,
      OPT_BACKEND          = 2,
      OPT_INPUT_LIST       = 3,
      OPT_OUTPUT_DIR       = 4,
      OPT_OP_PACKAGES      = 5,
      OPT_DEBUG_OUTPUTS    = 6,
      OPT_OUTPUT_DATA_TYPE = 7,
      OPT_INPUT_DATA_TYPE  = 8,
      OPT_LOG_LEVEL        = 9,
      OPT_PROFILING_LEVEL  = 10,
      OPT_RETRIEVE_CONTEXT = 11,
      OPT_SAVE_CONTEXT     = 12,
      OPT_VERSION          = 13,
      OPT_SYSTEM_LIBRARY   = 14
    };

    bool loadFromCachedBinary = false;

    // Command line parsing loop
    int longIndex = 0;
    int opt       = 0;
    std::string modelPath = "/qnn-projects/QNN-test-libs/example_libs/x86_64-linux-clang/libqnn_model_float.so";
    std::string backEndPath = "/qnn-projects/QNN-test-libs/libQnnCpu.so";
    std::string inputListPaths = "/qnn-projects/QNN-test-libs/input_list_float.txt";
    bool debug =  true;
    std::string outputPath;
    std::string opPackagePaths = "/qnn-projects/QNN-test-libs/libQnnCpuOpPackageExample.so:QnnOpPackage_interfaceProvider";
    iotensor::OutputDataType parsedOutputDataType   = iotensor::OutputDataType::FLOAT_ONLY;
    iotensor::InputDataType parsedInputDataType     = iotensor::InputDataType::FLOAT;
    sample_app::ProfilingLevel parsedProfilingLevel = ProfilingLevel::OFF;
    bool dumpOutputs                                = true;
    std::string cachedBinaryPath;
    std::string saveBinaryName;
    QnnLog_Level_t logLevel{QNN_LOG_LEVEL_ERROR};
    std::string systemLibraryPath;
    
    if (!modelPath.empty()) {
      if (!cachedBinaryPath.empty()) {
        std::exit(EXIT_FAILURE);
      }
    } else {
      if (cachedBinaryPath.empty()) {
        std::exit(EXIT_FAILURE);
      }
    }

    if (!cachedBinaryPath.empty() && !saveBinaryName.empty()) {
      std::exit(EXIT_FAILURE);
    }

    if (backEndPath.empty()) {
      std::exit(EXIT_FAILURE);
    }

    if (inputListPaths.empty()) {
      std::exit(EXIT_FAILURE);
    }

    if (loadFromCachedBinary && systemLibraryPath.empty()) {
      std::exit(EXIT_FAILURE);
    }

    QNN_INFO("Model: %s", modelPath.c_str());
    QNN_INFO("Backend: %s", backEndPath.c_str());

    QnnFunctionPointers qnnFunctionPointers;
    // Load backend and model .so and validate all the required function symbols are resolved
    auto statusCode = dynamicloadutil::getQnnFunctionPointers(backEndPath,
                                                              modelPath,
                                                              &qnnFunctionPointers,
                                                              &sg_backendHandle,
                                                              !loadFromCachedBinary,
                                                              &sg_modelHandle);
    
    if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
      if (dynamicloadutil::StatusCode::FAIL_LOAD_BACKEND == statusCode) {
        exitWithMessage(
            "Error initializing QNN Function Pointers: could not load backend: " + backEndPath,
            EXIT_FAILURE);
      } else if (dynamicloadutil::StatusCode::FAIL_LOAD_MODEL == statusCode) {
        exitWithMessage(
            "Error initializing QNN Function Pointers: could not load model: " + modelPath,
            EXIT_FAILURE);
      } else {
        exitWithMessage("Error initializing QNN Function Pointers", EXIT_FAILURE);
      }
    }

    if (loadFromCachedBinary) {
      statusCode =
          dynamicloadutil::getQnnSystemFunctionPointers(systemLibraryPath, &qnnFunctionPointers);
      if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
        exitWithMessage("Error initializing QNN System Function Pointers", EXIT_FAILURE);
      }
    }

    QnnBackendInitialize(qnnFunctionPointers,
                inputListPaths,
                opPackagePaths,
                sg_backendHandle,
                outputPath,
                debug,
                parsedOutputDataType,
                parsedInputDataType,
                parsedProfilingLevel,
                dumpOutputs,
                cachedBinaryPath,
                saveBinaryName);
}

void QNNBackend::init() {

  r_init();
}

void QNNBackend::release() {
  r_release();
}

int32_t QNNBackend::r_release() {
  QNNBackend* app = this;


  if (StatusCode::SUCCESS != app->freeContext()) {
    return app->reportError("Context Free failure");
  }

  auto devicePropertySupportStatus = app->isDevicePropertySupported();
  if (StatusCode::FAILURE != devicePropertySupportStatus) {
    auto freeDeviceStatus = app->freeDevice();
    if (StatusCode::SUCCESS != freeDeviceStatus) {
      return app->reportError("Device Free failure");
    }
  }
}

int32_t QNNBackend::r_init() {

    {
      bool loadFromCachedBinary{false};
      // std::unique_ptr<QnnSampleApp> app =
      //     processCommandLine(argc, argv, loadFromCachedBinary);

      QNNBackend* app = this;

      if (nullptr == app) {
        return EXIT_FAILURE;
      }

      QNN_INFO("qnn-sample-app build version: %s", getBuildId().c_str());
      QNN_INFO("Backend        build version: %s", getBackendBuildId().c_str());


      if (StatusCode::SUCCESS != app->initialize()) {
        return app->reportError("Initialization failure");
      }

      if (StatusCode::SUCCESS != app->initializeBackend()) {
        return app->reportError("Backend Initialization failure");
      }
      

      auto devicePropertySupportStatus = app->isDevicePropertySupported();
      if (StatusCode::FAILURE != devicePropertySupportStatus) {
        auto createDeviceStatus = app->createDevice();
        if (StatusCode::SUCCESS != createDeviceStatus) {
          return app->reportError("Device Creation failure");
        }
      }

      if (StatusCode::SUCCESS != app->initializeProfiling()) {
        return app->reportError("Profiling Initialization failure");
      }

      if (StatusCode::SUCCESS != app->registerOpPackages()) {
        return app->reportError("Register Op Packages failure");
      }

      if (!loadFromCachedBinary) {
        if (StatusCode::SUCCESS != app->createContext()) {
          return app->reportError("Context Creation failure");
        }
        if (StatusCode::SUCCESS != app->composeGraphs()) {
          return app->reportError("Graph Prepare failure");
        }
        if (StatusCode::SUCCESS != app->finalizeGraphs()) {
          return app->reportError("Graph Finalize failure");
        }
      } else {
        if (StatusCode::SUCCESS != app->createFromBinary()) {
          return app->reportError("Create From Binary failure");
        }
      }

      if (StatusCode::SUCCESS != app->executeGraphs()) {
        return app->reportError("Graph Execution failure");
      }

    }

    
}

void QNNBackend::registerOps() {
    addCreator(ADD, (QNNBackend::Creator *)new QNNAddCreator());
}

std::string QNNBackend::getBackendBuildId() {
  char* backendBuildId{nullptr};
  if (QNN_SUCCESS !=
      m_qnnFunctionPointers.qnnInterface.backendGetBuildId((const char**)&backendBuildId)) {
    QNN_ERROR("Unable to get build Id from the backend.");
  }
  return (backendBuildId == nullptr ? std::string("") : std::string(backendBuildId));
}

// --------- temp dev functions to test QNNBackend
int32_t QNNBackend::graphInitialize() {
    QNN_INFO("qnn-sample-app build version: %s", getBuildId().c_str());
    QNN_INFO("Backend        build version: %s", getBackendBuildId().c_str());

    if (StatusCode::SUCCESS != this->initialize()) {
        return this->reportError("Initialization failure");
    }

    if (StatusCode::SUCCESS != this->initializeBackend()) {
        return this->reportError("Backend Initialization failure");
    }

    auto devicePropertySupportStatus = this->isDevicePropertySupported();
    if (StatusCode::FAILURE != devicePropertySupportStatus) {
        auto createDeviceStatus = this->createDevice();
        if (StatusCode::SUCCESS != createDeviceStatus) {
            return this->reportError("Device Creation failure");
        }
    }

    if (StatusCode::SUCCESS != this->initializeProfiling()) {
        return this->reportError("Profiling Initialization failure");
    }

    if (StatusCode::SUCCESS != this->registerOpPackages()) {
        return this->reportError("Register Op Packages failure");
    }

    if (StatusCode::SUCCESS != this->createContext()) {
        return this->reportError("Context Creation failure");
    }
    
    // initialize graph info, set graph info, graph count
    // acting the same as composeGraphs
    const QnnGraph_Config_t **graphConfigs = nullptr;
    qnn_wrapper_api::ModelError_t err = qnn_wrapper_api::MODEL_NO_ERROR;
    VALIDATE(qnn_wrapper_api::getQnnGraphConfigFromInfo(
                 "convReluModel", (const qnn_wrapper_api::GraphConfigInfo_t **)m_graphConfigsInfo, 1, graphConfigs),
             err);
    VALIDATE(qnnModel.initialize(m_backendHandle,
                                      m_qnnFunctionPointers.qnnInterface,
                                      m_context,
                                      "mllmQnnModel",
                                      m_debug,
                                      DO_GRAPH_NODE_VALIDATIONS,
                                      graphConfigs),
             err);
    // TODO: err should not be converted to int32_t directly
    return 0;
}

qnn_wrapper_api::ModelError_t QNNBackend::graphAddNode(string name,
                                                       string nodeType,
                                                       std::vector<const char *> inputTensorNames,
                                                       std::vector<Qnn_Tensor_t> outputTensors,
                                                       string packageName) {
    qnn_wrapper_api::ModelError_t err = qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR;
    VALIDATE(qnnModel.addNode(
                 QNN_OPCONFIG_VERSION_1,  // Op_Config_t Version
                 name.c_str(),            // Node Name
                 packageName.c_str(),     // Package Name
                 nodeType.c_str(),        // Qnn Node Type
                 nullptr,                 // Node Params
                 0,                       // Num Node Params
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
    uint32_t numModels = 1;
    m_graphsCount = 1;
    // Populate the constructed graphs in provided output variables
    qnn_wrapper_api::ModelError_t err = qnn_wrapper_api::MODEL_NO_ERROR;
    VALIDATE(getGraphInfoFromModels(*models, numModels, &m_graphsInfo), err);

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
// ---------

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
  // print m_inputFileLists
  for (auto const& inputFileList : m_inputFileLists) {
    std::cout << "inputFileList: " << inputFileList.size() << std::endl;
    for(auto& inputFile : inputFileList) {
      std::cout << "inputFile: " << inputFile.size() << std::endl;
      std::cout << "inputFile: " << inputFile.front() << std::endl;
      std::cout << "--------------" << std::endl;
    }
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
int32_t QNNBackend::reportError(const std::string& err) {
  QNN_ERROR("%s", err.c_str());
  return EXIT_FAILURE;
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
  auto returnStatus = StatusCode::SUCCESS;
  if (!m_saveBinaryName.empty()) {
    QNN_INFO("Before saveBinary(): saving context and metadata.");
    returnStatus = saveBinary();
  } else {
    QNN_DEBUG("m_saveBinaryName is empty()");
  }
  return returnStatus;
}

StatusCode QNNBackend::createFromBinary() {
  if (m_cachedBinaryPath.empty()) {
    QNN_ERROR("No name provided to read binary file from.");
    return StatusCode::FAILURE;
  }
  if (nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate ||
      nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo ||
      nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextFree) {
    QNN_ERROR("QNN System function pointers are not populated.");
    return StatusCode::FAILURE;
  }
  uint64_t bufferSize{0};
  std::shared_ptr<uint8_t> buffer{nullptr};
  // read serialized binary into a byte buffer
  tools::datautil::StatusCode status{tools::datautil::StatusCode::SUCCESS};
  std::tie(status, bufferSize) = tools::datautil::getFileSize(m_cachedBinaryPath);
  if (0 == bufferSize) {
    QNN_ERROR("Received path to an empty file. Nothing to deserialize.");
    return StatusCode::FAILURE;
  }
  buffer = std::shared_ptr<uint8_t>(new uint8_t[bufferSize], std::default_delete<uint8_t[]>());
  if (!buffer) {
    QNN_ERROR("Failed to allocate memory.");
    return StatusCode::FAILURE;
  }

  status = tools::datautil::readBinaryFromFile(
      m_cachedBinaryPath, reinterpret_cast<uint8_t*>(buffer.get()), bufferSize);
  if (status != tools::datautil::StatusCode::SUCCESS) {
    QNN_ERROR("Failed to read binary data.");
    return StatusCode::FAILURE;
  }

  // inspect binary info
  auto returnStatus = StatusCode::SUCCESS;
  QnnSystemContext_Handle_t sysCtxHandle{nullptr};
  if (QNN_SUCCESS != m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
    QNN_ERROR("Could not create system handle.");
    returnStatus = StatusCode::FAILURE;
  }
  const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
  Qnn_ContextBinarySize_t binaryInfoSize{0};
  if (StatusCode::SUCCESS == returnStatus &&
      QNN_SUCCESS != m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo(
                         sysCtxHandle,
                         static_cast<void*>(buffer.get()),
                         bufferSize,
                         &binaryInfo,
                         &binaryInfoSize)) {
    QNN_ERROR("Failed to get context binary info");
    returnStatus = StatusCode::FAILURE;
  }

  // fill GraphInfo_t based on binary info
  if (StatusCode::SUCCESS == returnStatus &&
      !copyMetadataToGraphsInfo(binaryInfo, m_graphsInfo, m_graphsCount)) {
    QNN_ERROR("Failed to copy metadata.");
    returnStatus = StatusCode::FAILURE;
  }
  m_qnnFunctionPointers.qnnSystemInterface.systemContextFree(sysCtxHandle);
  sysCtxHandle = nullptr;

  if (StatusCode::SUCCESS == returnStatus &&
      nullptr == m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary) {
    QNN_ERROR("contextCreateFromBinaryFnHandle is nullptr.");
    returnStatus = StatusCode::FAILURE;
  }
  if (StatusCode::SUCCESS == returnStatus &&
      m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary(
          m_backendHandle,
          m_deviceHandle,
          (const QnnContext_Config_t**)&m_contextConfig,
          static_cast<void*>(buffer.get()),
          bufferSize,
          &m_context,
          m_profileBackendHandle)) {
    QNN_ERROR("Could not create context from binary.");
    returnStatus = StatusCode::FAILURE;
  }
  if (ProfilingLevel::OFF != m_profilingLevel) {
    extractBackendProfilingInfo(m_profileBackendHandle);
  }
  m_isContextCreated = true;
  if (StatusCode::SUCCESS == returnStatus) {
    for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
      if (nullptr == m_qnnFunctionPointers.qnnInterface.graphRetrieve) {
        QNN_ERROR("graphRetrieveFnHandle is nullptr.");
        returnStatus = StatusCode::FAILURE;
        break;
      }
      if (QNN_SUCCESS !=
          m_qnnFunctionPointers.qnnInterface.graphRetrieve(
              m_context, (*m_graphsInfo)[graphIdx].graphName, &((*m_graphsInfo)[graphIdx].graph))) {
        QNN_ERROR("Unable to retrieve graph handle for graph Idx: %d", graphIdx);
        returnStatus = StatusCode::FAILURE;
      }
    }
  }
  if (StatusCode::SUCCESS != returnStatus) {
    QNN_DEBUG("Cleaning up graph Info structures.");
    qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
  }
  return returnStatus;
}

StatusCode QNNBackend::saveBinary() {
  if (m_saveBinaryName.empty()) {
    QNN_ERROR("No name provided to save binary file.");
    return StatusCode::FAILURE;
  }
  if (nullptr == m_qnnFunctionPointers.qnnInterface.contextGetBinarySize ||
      nullptr == m_qnnFunctionPointers.qnnInterface.contextGetBinary) {
    QNN_ERROR("contextGetBinarySizeFnHandle or contextGetBinaryFnHandle is nullptr.");
    return StatusCode::FAILURE;
  }
  uint64_t requiredBufferSize{0};
  if (QNN_CONTEXT_NO_ERROR !=
      m_qnnFunctionPointers.qnnInterface.contextGetBinarySize(m_context, &requiredBufferSize)) {
    QNN_ERROR("Could not get the required binary size.");
    return StatusCode::FAILURE;
  }
  std::unique_ptr<uint8_t[]> saveBuffer(new uint8_t[requiredBufferSize]);
  if (nullptr == saveBuffer) {
    QNN_ERROR("Could not allocate buffer to save binary.");
    return StatusCode::FAILURE;
  }
  uint64_t writtenBufferSize{0};
  if (QNN_CONTEXT_NO_ERROR !=
      m_qnnFunctionPointers.qnnInterface.contextGetBinary(m_context,
                                                          reinterpret_cast<void*>(saveBuffer.get()),
                                                          requiredBufferSize,
                                                          &writtenBufferSize)) {
    QNN_ERROR("Could not get binary.");
    return StatusCode::FAILURE;
  }
  if (requiredBufferSize < writtenBufferSize) {
    QNN_ERROR(
        "Illegal written buffer size [%d] bytes. Cannot exceed allocated memory of [%d] bytes",
        writtenBufferSize,
        requiredBufferSize);
    return StatusCode::FAILURE;
  }
  auto dataUtilStatus = tools::datautil::writeBinaryToFile(
      m_outputPath, m_saveBinaryName + ".bin", (uint8_t*)saveBuffer.get(), writtenBufferSize);
  if (tools::datautil::StatusCode::SUCCESS != dataUtilStatus) {
    QNN_ERROR("Error while writing binary to file.");
    return StatusCode::FAILURE;
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


}