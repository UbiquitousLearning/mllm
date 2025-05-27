#include <cstdint>
#include <inttypes.h>

#include <cstring>
#include <iostream>
#include <memory>

#include "Log.h"
#include "Module.hpp"
#include "OpDefined.hpp"
#include "QNNBackend.hpp"
#include "ParamLoader.hpp"
#include "QnnModel.hpp"
#include "Utils/QnnSampleAppUtils.hpp"
#include "Utils/IOTensor.hpp"
#include "Utils/DynamicLoadUtil.hpp"
#include "Log/Logger.hpp"
#include "WrapperUtils/QnnWrapperUtils.hpp"
#include "QNNMemoryManager.hpp"
#include "QnnTypes.h"
#include "HTP/QnnHtpGraph.h"
#include "Layer.hpp"

#include "Types.hpp"
#include "op/QNNAdd.hpp"
#include "op/QNNCausalMask.hpp"
#include "op/QNNGELU.hpp"
#include "op/QNNLinearINT8.hpp"
#include "op/QNNMatmul.hpp"
#include "op/QNNMul.hpp"
#include "op/QNNLayerNorm.hpp"
#include "op/QNNRMSNorm.hpp"
#include "op/QNNRoPE.hpp"
#include "op/QNNScale.hpp"
#include "op/QNNSiLU.hpp"
#include "op/QNNSoftMax.hpp"
#include "op/QNNView.hpp"
#include "op/QNNReLU.hpp"
#include "op/QNNQuantize.hpp"
#include "op/QNNDequantize.hpp"
#include "op/QNNMergeOutput.hpp"
#include "op/QNNSplitInput.hpp"
#include "op/QNNTranspose.hpp"
#include "op/QNNSuperSiLU.hpp"
#include "op/QNNIRoPE.hpp"

#include "memory/MemInspect.hpp"

#ifdef DEBUGPRINT
#include "Timing.hpp"
#endif

using namespace qnn;
using namespace qnn::tools;
using namespace qnn::tools::sample_app;

// Flag to determine if Backend should node validation for each opNode added
#ifdef QNN_VALIDATE_NODE
#define DO_GRAPH_NODE_VALIDATIONS 1
#else
#define DO_GRAPH_NODE_VALIDATIONS 0
#endif

namespace mllm {

void QNNBackend::registerOps() {
    addCreator(ADD, (QNNBackend::Creator *)new QNNAddCreator());
    addCreator(CAUSALMASK, (QNNBackend::Creator *)(new QNNCausalMaskCreator()));
    addCreator(MATMUL, (QNNBackend::Creator *)(new QNNMatmulCreator()));
    addCreator(RMSNORM, (QNNBackend::Creator *)(new QNNRMSNormCreator()));
    addCreator(LAYERNORM, (QNNBackend::Creator *)(new QNNLayerNormCreator()));
    addCreator(ROPE, (QNNBackend::Creator *)(new QNNRoPECreator()));
    addCreator(IROPE, (QNNBackend::Creator *)(new QNNIRoPECreator()));
    addCreator(SCALE, (QNNBackend::Creator *)(new QNNScaleCreator()));
    addCreator(SILU, (QNNBackend::Creator *)(new QNNSiLUCreator()));
    addCreator(SOFTMAX, (QNNBackend::Creator *)(new QNNSoftMaxCreator()));
    addCreator(LINEAR, (QNNBackend::Creator *)(new QNNLinearINT8Creator()));
    addCreator(LINEARINT8, (QNNBackend::Creator *)(new QNNLinearINT8Creator()));
    addCreator(MUL, (QNNBackend::Creator *)(new QNNMulCreator()));
    addCreator(VIEW, (QNNBackend::Creator *)(new QNNViewCreator()));
    addCreator(RELU, (QNNBackend::Creator *)(new QNNReLUCreator()));
    addCreator(OP_GELU, (QNNBackend::Creator *)(new QNNGELUCreator()));
    addCreator(QUANTIZE, (QNNBackend::Creator *)(new QNNQuantizeCreator()));
    addCreator(DEQUANTIZE, (QNNBackend::Creator *)(new QNNDequantizeCreator()));
    addCreator(MERGEOUTPUT, (QNNBackend::Creator *)(new QNNMergeOutputCreator()));
    addCreator(SPLITINPUT, (QNNBackend::Creator *)(new QNNSplitInputCreator()));
    addCreator(TRANSPOSE, (QNNBackend::Creator *)(new QNNTransposeCreator()));
    addCreator(SUPERSILU, (QNNBackend::Creator *)(new QNNSuperSiLUCreator()));
}

QNNBackend::QNNBackend(shared_ptr<MemoryManager> mm) :
    Backend(mm) {
    type_ = BackendType::MLLM_QNN; // used in Tensor.device()
    if (!log::initializeLogging()) {
        MLLM_LOG_ERROR_STREAM << "ERROR: Unable to initialize logging!\n";
        return;
    }
    // TODO: make debug level configuable
    log::setLogLevel(QnnLog_Level_t::QNN_LOG_LEVEL_ERROR);

    std::string backEndPath = "libQnnHtp.so";
    std::string opPackagePaths = "libQnnLLaMAPackage_CPU.so:LLaMAPackageInterfaceProvider:CPU,libQnnLLaMAPackage_HTP.so:LLaMAPackageInterfaceProvider:HTP";

    // TODO: make these configuable
    m_debug = false; // when set true, NATIVE tensor will be regared as APP_READ tensor
    m_inputDataType = iotensor::InputDataType::NATIVE;
    m_profilingLevel = ProfilingLevel::OFF;

    m_isBackendInitialized = false;
    m_isContextCreated = false;

    // config path strings
    split(m_opPackagePaths, opPackagePaths, ',');

    if (backEndPath.empty()) {
        std::exit(EXIT_FAILURE);
    }
    MLLM_LOG_INFO_LEGACY("Backend: %s", backEndPath.c_str());

    // Load backend and validate all the required function symbols are resolved
    auto statusCode = dynamicloadutil::getQnnFunctionPointers(backEndPath,
                                                              "",
                                                              &m_qnnFunctionPointers,
                                                              &m_backendLibraryHandle,
                                                              false,
                                                              nullptr);
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

    // init qnn resources
    {
        MLLM_LOG_INFO_LEGACY("Backend        build version: %s", getBackendBuildId().c_str());

        // initialize logging in the backend
        if (log::isLogInitialized()) {
            auto logCallback = log::getLogCallback();
            auto logLevel = log::getLogLevel();
            // MLLM_LOG_INFO("Initializing logging in the backend. Callback: {}, Log Level: {}",
            //               logCallback,
            //               logLevel);
            if (QNN_SUCCESS != m_qnnFunctionPointers.qnnInterface.logCreate(logCallback, logLevel, &m_logHandle)) {
                MLLM_LOG_WARN_LEGACY("Unable to initialize logging in the backend.");
            }
        } else {
            MLLM_LOG_WARN_LEGACY("Logging not available in the backend.");
        }

        // initialize QnnBackend
        auto qnnStatus = m_qnnFunctionPointers.qnnInterface.backendCreate(
            m_logHandle, (const QnnBackend_Config_t **)m_backendConfig, &m_backendHandle);
        if (QNN_BACKEND_NO_ERROR != qnnStatus) {
            MLLM_LOG_ERROR("Could not initialize backend due to error = {}", (unsigned int)qnnStatus);
            this->reportError("Backend Initialization failure");
        }
        MLLM_LOG_INFO("Initialize Backend Returned Status = {}", (unsigned int)qnnStatus);
        m_isBackendInitialized = true;

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
    }

    // register ops
    this->registerOps();
}

QNNBackend::~QNNBackend() {
    terminateBackend();
    // free creaters in map_creator_
    for (auto &iter : map_creator_) {
        delete iter.second;
    }
    // free qnn backend resource
    auto devicePropertySupportStatus = this->isDevicePropertySupported();
    if (StatusCode::FAILURE != devicePropertySupportStatus) {
        auto freeDeviceStatus = this->freeDevice();
        if (StatusCode::SUCCESS != freeDeviceStatus) {
            this->reportError("Device Free failure");
        }
    }
    // free dynamic library handle
    if (m_backendLibraryHandle) {
        pal::dynamicloading::dlClose(m_backendLibraryHandle);
    }
    QNN_INFO("Free handle");
}

void QNNBackend::onSetUpStart(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName) {
    auto returnStatus = StatusCode::SUCCESS;

    // create a new graph
    qnnModelIndex_ = qnnModels_.size();
    qnnModelIndexMap_.insert(std::make_pair(graphName, qnnModelIndex_));
    qnnModels_.push_back(qnn_wrapper_api::QnnModel());
    // create qnn context, assign context to qnn memory manager
    if (StatusCode::SUCCESS != this->createContext()) {
        this->reportError("Context Creation failure");
    }
#ifdef QNN_ARM
    auto qnnMM = std::static_pointer_cast<QNNMemoryManager>(mem_manager_);
    qnnMM->setQnnInterfaceAndContext(m_context);
#endif

    // initialize qnn graph info, set graph info, graph count
    // NOTE: currently not using it
    QnnHtpGraph_CustomConfig_t customConfig;
    // customConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
    // customConfig.numHvxThreads = 4; // set a number. MAX = number of HVX HW blocks for that SoC
    customConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
    customConfig.vtcmSizeInMB = 8;

    QnnGraph_Config_t graphConfig;
    graphConfig.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graphConfig.customConfig = &customConfig;

    const QnnGraph_Config_t *pGraphConfig[] = {&graphConfig, NULL};

    const QnnGraph_Config_t **graphConfigs = pGraphConfig;

    m_graphConfigsInfoCount = 1;

    qnn_wrapper_api::ModelError_t err = qnn_wrapper_api::getQnnGraphConfigFromInfo(
        graphName.c_str(), (const qnn_wrapper_api::GraphConfigInfo_t **)m_graphConfigsInfo, m_graphConfigsInfoCount, graphConfigs);
    if (err != qnn_wrapper_api::MODEL_NO_ERROR) {
        this->reportError("Graph Config Info failure");
    }

    err = qnnModels_[qnnModelIndex_].initialize(m_backendHandle,
                                                m_qnnFunctionPointers.qnnInterface,
                                                m_context,
                                                graphName.c_str(),
                                                m_debug,
                                                DO_GRAPH_NODE_VALIDATIONS,
                                                graphConfigs);
    if (err != qnn_wrapper_api::MODEL_NO_ERROR) {
        this->reportError("Graph Initialization failure: " + graphName);
    }

    // To avoid no input, we put inputs here.
    // For splitinput op input, the seq will be divided as 5, and we add the input in split ops.
    for (auto &input : inputs) {
        Qnn_DataType_t data_type;
        auto quantizeDefined = QNN_DEFINITION_UNDEFINED;
        auto quantizeType = QNN_QUANTIZATION_ENCODING_UNDEFINED;
        float scale = 0.0f;
        AbstructLoader *loader = nullptr;
        if (Module::llm_model_ptr == nullptr) { // old frontend
            loader = dataLoader_;
        } else { // new frontend
            loader = Module::llm_model_ptr->loader;
        }
        Tensor scaleTensor(this);
        scaleTensor.reshape(1, 1, 1, 1);
        scaleTensor.setDtype(MLLM_TYPE_F32);
        scaleTensor.alloc();

        switch (input->dtype()) {
        case MLLM_TYPE_F32:
            data_type = QNN_DATATYPE_FLOAT_32;
            break;
        case MLLM_TYPE_I8: {
            data_type = QNN_DATATYPE_SFIXED_POINT_8;
            quantizeDefined = QNN_DEFINITION_DEFINED;
            quantizeType = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;

            string scaleName = input->name();

            std::string wordToRemove = "outtensor-";
            int pos = scaleName.find(wordToRemove);
            if (pos != -1) { // old frontend merge/split generated tensor
                scaleName = scaleName.substr(wordToRemove.length());
                wordToRemove = "or_split";
                if (scaleName.find(wordToRemove) != -1) {
                    pos = scaleName.find("or_split");
                    // scaleName.erase(pos, wordToRemove.length());
                    scaleName = scaleName.substr(0, pos);
                    // o
                    scaleName += "o_proj.input_scale";
                } else if (scaleName.find("ires_split") != -1) {
                    pos = scaleName.find("ires_split");
                    wordToRemove = "ires_split";
                    // scaleName.erase(pos, wordToRemove.length());
                    scaleName = scaleName.substr(0, pos);
                    // q
                    scaleName += "q_proj.input_scale";
                } else if (scaleName.find("fres_split") != -1) {
                    pos = scaleName.find("fres_split");
                    wordToRemove = "fres_split";
                    // scaleName.erase(pos, wordToRemove.length());
                    scaleName = scaleName.substr(0, pos);
                    // fc1
                    scaleName += "up_proj.input_scale";
                }
            } else { // new frontend no merge/split condition
                std::string prefix = "out-", suffix = ".quantize";
                if (input->name().find(prefix) != std::string::npos) {
                    scaleName = input->name().substr(prefix.length());
                }
                if (scaleName.find(suffix) != std::string::npos) {
                    scaleName = scaleName.substr(0, scaleName.length() - suffix.length());
                }
                scaleName += ".input_scale";
            }
            scaleTensor.setName(scaleName);
            loader->load(&scaleTensor);
            scale = roundf(scaleTensor.hostPtr<float>()[0] / 127.0 * 100000) / 100000;
            scaleTensor.free();

            break;
        }
        default:
            MLLM_LOG_ERROR_STREAM << "[ERROR] QNNBackend not support dtype: " << input->dtype() << std::endl;
            data_type = QNN_DATATYPE_FLOAT_32;
        }

        uint32_t dimensionsInput[4] = {
            static_cast<uint32_t>(input->batch()),
            static_cast<uint32_t>(input->sequence()),
            static_cast<uint32_t>(input->head()),
            static_cast<uint32_t>(input->dimension()),
        };

        qnnModels_[qnnModelIndex_].addTensor(input->name().c_str(),
                                             (Qnn_Tensor_t){
                                                 .version = QNN_TENSOR_VERSION_1,
                                                 .v1 = {
                                                     .id = 0,
                                                     .name = input->name().c_str(),
                                                     .type = QNN_TENSOR_TYPE_APP_WRITE,
                                                     .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                     .dataType = data_type,
                                                     .quantizeParams = {quantizeDefined,
                                                                        quantizeType,
                                                                        {.scaleOffsetEncoding = {.scale = scale, .offset = 0}}},
                                                     .rank = 4,
                                                     .dimensions = dimensionsInput,
                                                     .memType = QNN_TENSORMEMTYPE_RAW,
                                                     .clientBuf = {.data = nullptr,
                                                                   .dataSize = 0}}});
    }

    // create a new inputBuffer and outputBuffer for the graph
    inputBufferMap.insert(std::make_pair(graphName, std::vector<uint8_t *>(inputs.size())));
    outputBufferMap.insert(std::make_pair(graphName, std::vector<uint8_t *>()));

    currentInputBuffers = &inputBufferMap[graphName];
    currentOutputBuffers = &outputBufferMap[graphName];

    // push input tensors to the buffer list
    for (int i = 0; i < inputs.size(); i++) {
        (*currentInputBuffers)[i] = inputs[i]->hostPtr<uint8_t>();
    }
}

void QNNBackend::onSetUpEnd(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName) {
    currentInputBuffers = &inputBufferMap[graphName];
    currentOutputBuffers = &outputBufferMap[graphName];
    qnnModelIndex_ = qnnModelIndexMap_[graphName];
    PRINT_MEMORY_USAGE("before graph finilize")
    auto status = graphFinilize();
    PRINT_MEMORY_USAGE("after graph finilize")
    if (qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR != status) {
        this->reportError("Graph Finalization failure");
    }

    auto returnStatus = StatusCode::SUCCESS;

    Qnn_Tensor_t *qnnInputs = nullptr;
    Qnn_Tensor_t *qnnOutputs = nullptr;

    auto graphInfo = graphInfoMap_[qnnModelIndex_];

    // directly get qnnInputs and qnnOutputs from graphInfo.outputTensors
    if (iotensor::StatusCode::SUCCESS != m_ioTensor.setupInputAndOutputTensors(&qnnInputs, &qnnOutputs, *graphInfo)) {
        MLLM_LOG_ERROR_LEGACY("Error in setting up Input and output Tensors for qnnModelIndex_: %d", qnnModelIndex_);
        returnStatus = StatusCode::FAILURE;
    }

    auto qnnMM = std::static_pointer_cast<QNNMemoryManager>(mem_manager_);

    // register input and output tensor to qnn shared buffers
    // must insure the inputs and outputs of mllm graph are the same as the qnn graph
    // op created io tensors (kvcache, wnop...) should be solved
#ifdef DEBUGPRINT
    std::cout << "input tensors num:" << graphInfo->numInputTensors << std::endl;
    std::cout << "output tensors num:" << graphInfo->numOutputTensors << std::endl;
#endif

    for (int i = 0; i < graphInfo->numInputTensors; i++) {
        qnnMM->registerQnnTensor((*currentInputBuffers)[i], qnnInputs[i]);
#ifdef DEBUGPRINT
        if (i < inputs.size()) {
            std::cout << "\nregistered input tensor: " << inputs[i]->hostPtr<void>() << " backend staged ptr: " << (void *)(*currentInputBuffers)[i] << std::endl;
        } else {
            std::cout << "\n registered op added input" << std::endl;
        }
        std::cout << "qnn input tensor name: " << qnnInputs[i].v1.name << std::endl;
        std::cout << "qnn input tensor scale: " << qnnInputs[i].v1.quantizeParams.scaleOffsetEncoding.scale << std::endl;
#endif
    }
    for (int i = 0; i < graphInfo->numOutputTensors; i++) {
        qnnMM->registerQnnTensor((*currentOutputBuffers)[i], qnnOutputs[i]);
#ifdef DEBUGPRINT
        if (i < outputs.size()) {
            std::cout << "\nregistered output tensor: " << outputs[i]->hostPtr<void>() << " backend staged ptr: " << (void *)(*currentOutputBuffers)[i] << std::endl;
        } else {
            std::cout << "\n registered op added output" << std::endl;
        }
        std::cout << "qnn output tensor name: " << qnnOutputs[i].v1.name << std::endl;
        std::cout << "qnn output tensor scale: " << qnnOutputs[i].v1.quantizeParams.scaleOffsetEncoding.scale << std::endl;
#endif
    }

    inputsMap_[qnnModelIndex_] = qnnInputs;
    outputsMap_[qnnModelIndex_] = qnnOutputs;
}

void QNNBackend::onExecuteStart(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName) {
    // to support multi-thread, we need local variable.
    // update currentInputBuffers, currentOutputBuffers, qnnModelIndex_
    auto t_qnnModelIndex_ = qnnModelIndexMap_[graphName];

    qnn_wrapper_api::GraphInfo_t *graphInfo = graphInfoMap_[t_qnnModelIndex_];

    Qnn_Tensor_t *inputs_ = inputsMap_[t_qnnModelIndex_];
    Qnn_Tensor_t *outputs_ = outputsMap_[t_qnnModelIndex_];

    Qnn_ErrorHandle_t executeStatus = QNN_GRAPH_NO_ERROR;
#ifdef DEBUGPRINT
    uint64_t t_start = mllm_time_us();
#endif
    executeStatus =
        m_qnnFunctionPointers.qnnInterface.graphExecute(graphInfo->graph,
                                                        inputs_,
                                                        graphInfo->numInputTensors,
                                                        outputs_,
                                                        graphInfo->numOutputTensors,
                                                        m_profileBackendHandle,
                                                        nullptr);
#ifdef DEBUGPRINT
    uint64_t t_end = mllm_time_us();
    std::cout << "QNN execution time " << (t_end - t_start) / 1000.0F << " ms" << std::endl;
#endif

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
        MLLM_LOG_ERROR_STREAM << "Error in executing graph: " << graphName << std::endl;
    }

    if (ProfilingLevel::OFF != m_profilingLevel) {
        extractBackendProfilingInfo(m_profileBackendHandle);
    }
}

void QNNBackend::onExecuteEnd(std::vector<std::shared_ptr<Tensor>> &outputs, const string &graph_name) {
}

void QNNBackend::freeGraphDataStructure(string graphName) {
    auto it = qnnModelIndexMap_.find(graphName);
    if (it != qnnModelIndexMap_.end()) {
        qnnModelIndex_ = it->second;

        qnnModels_[qnnModelIndex_].freeTensors();
        qnnModels_[qnnModelIndex_].clearGraph();
    }

    inputBufferMap[graphName].resize(0);
    outputBufferMap[graphName].resize(0);
}

void QNNBackend::afterAllGraphsExecute() {
    // clear old models.
    qnnModelIndexMap_.clear();

    auto qnnMM = std::static_pointer_cast<QNNMemoryManager>(mem_manager_);
    qnnMM->deRegisterQnnTensor();

    this->freeContext();

    inputBufferMap.clear();
    outputBufferMap.clear();

    graphInfoMap_.clear();
    inputsMap_.clear();
    outputsMap_.clear();
}

std::string QNNBackend::getBackendBuildId() {
    char *backendBuildId{nullptr};
    if (QNN_SUCCESS != m_qnnFunctionPointers.qnnInterface.backendGetBuildId((const char **)&backendBuildId)) {
        MLLM_LOG_ERROR_LEGACY("Unable to get build Id from the backend.");
    }
    return (backendBuildId == nullptr ? std::string("") : std::string(backendBuildId));
}

qnn_wrapper_api::ModelError_t QNNBackend::graphAddNode(string name,
                                                       string nodeType,
                                                       std::vector<string> inputTensorNames,
                                                       std::vector<Qnn_Tensor_t> outputTensors,
                                                       std::vector<Qnn_Param_t> params,
                                                       string packageName) {
    qnn_wrapper_api::ModelError_t err = qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR;
    Qnn_Param_t *paramsPtr = nullptr;
    if (!params.empty()) {
        paramsPtr = params.data();
    }
    VALIDATE(qnnModels_[qnnModelIndex_].addNode(
                 QNN_OPCONFIG_VERSION_1,  // Op_Config_t Version
                 name.c_str(),            // Node Name
                 packageName.c_str(),     // Package Name
                 nodeType.c_str(),        // Qnn Node Type
                 paramsPtr,               // Node Params
                 params.size(),           // Num Node Params
                 inputTensorNames,        // Input Tensor Names
                 inputTensorNames.size(), // Num Input Tensor Names
                 outputTensors.data(),    // Output Tensors
                 outputTensors.size()     // Num Output Tensors
                 ),
             err);
    return err;
}

qnn_wrapper_api::ModelError_t QNNBackend::graphFinilize() {
    // Populate the constructed graphs in provided output variables
    qnn_wrapper_api::ModelError_t err = qnn_wrapper_api::MODEL_NO_ERROR;
    qnn_wrapper_api::GraphInfo_t *graphInfo = nullptr;

    VALIDATE(getSingleGraphInfoFromModel(qnnModels_[qnnModelIndex_], &graphInfo), err);

    // Graph finalize
    if (QNN_GRAPH_NO_ERROR != m_qnnFunctionPointers.qnnInterface.graphFinalize(graphInfo->graph, m_profileBackendHandle, nullptr)) {
        return qnn_wrapper_api::ModelError_t::MODEL_GRAPH_ERROR;
    }
    if (ProfilingLevel::OFF != m_profilingLevel) {
        extractBackendProfilingInfo(m_profileBackendHandle);
    }

    graphInfoMap_[qnnModelIndex_] = graphInfo;

    return qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR;
}

qnn_wrapper_api::ModelError_t QNNBackend::modelAddTensor(std::string nodeName, Qnn_Tensor_t tensor) {
    return qnnModels_[qnnModelIndex_].addTensor(nodeName.c_str(), tensor);
}

StatusCode QNNBackend::initializeProfiling() {
    if (ProfilingLevel::OFF != m_profilingLevel) {
        MLLM_LOG_INFO_LEGACY("Profiling turned on; level = %d", (int)m_profilingLevel);
        if (ProfilingLevel::BASIC == m_profilingLevel) {
            MLLM_LOG_INFO_LEGACY("Basic profiling requested. Creating Qnn Profile object.");
            if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileCreate(m_backendHandle, QNN_PROFILE_LEVEL_BASIC, &m_profileBackendHandle)) {
                MLLM_LOG_WARN_LEGACY("Unable to create profile handle in the backend.");
                return StatusCode::FAILURE;
            }
        } else if (ProfilingLevel::DETAILED == m_profilingLevel) {
            MLLM_LOG_INFO_LEGACY("Detailed profiling requested. Creating Qnn Profile object.");
            if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileCreate(m_backendHandle, QNN_PROFILE_LEVEL_DETAILED, &m_profileBackendHandle)) {
                MLLM_LOG_ERROR_LEGACY("Unable to create profile handle in the backend.");
                return StatusCode::FAILURE;
            }
        }
    }
    return StatusCode::SUCCESS;
}

// Simple method to report error from app to lib.
void QNNBackend::reportError(const std::string &err) {
    MLLM_LOG_ERROR_LEGACY("%s", err.c_str());
    exit(1);
}

// Terminate the backend after done.
StatusCode QNNBackend::terminateBackend() {
    if ((m_isBackendInitialized && nullptr != m_qnnFunctionPointers.qnnInterface.backendFree) && QNN_BACKEND_NO_ERROR != m_qnnFunctionPointers.qnnInterface.backendFree(m_backendHandle)) {
        MLLM_LOG_ERROR_LEGACY("Could not terminate backend");
        return StatusCode::FAILURE;
    }
    m_isBackendInitialized = false;
    return StatusCode::SUCCESS;
}

// Register op packages and interface providers supplied during
// object creation. If there are multiple op packages, register
// them sequentially in the order provided.
StatusCode QNNBackend::registerOpPackages() {
    const size_t pathIdx = 0;
    const size_t interfaceProviderIdx = 1;
    for (auto const &opPackagePath : m_opPackagePaths) {
        std::vector<std::string> opPackage;
        split(opPackage, opPackagePath, ':');
        QNN_DEBUG("opPackagePath: %s", opPackagePath.c_str());
        const char *target = nullptr;
        const size_t targetIdx = 2;
        if (opPackage.size() != 2 && opPackage.size() != 3) {
            MLLM_LOG_ERROR_LEGACY("Malformed opPackageString provided: %s", opPackagePath.c_str());
            return StatusCode::FAILURE;
        }
        if (opPackage.size() == 3) {
            target = (char *)opPackage[targetIdx].c_str();
        }
        if (nullptr == m_qnnFunctionPointers.qnnInterface.backendRegisterOpPackage) {
            MLLM_LOG_ERROR_LEGACY("backendRegisterOpPackageFnHandle is nullptr.");
            return StatusCode::FAILURE;
        }
        if (QNN_BACKEND_NO_ERROR != m_qnnFunctionPointers.qnnInterface.backendRegisterOpPackage(m_backendHandle, (char *)opPackage[pathIdx].c_str(), (char *)opPackage[interfaceProviderIdx].c_str(), target)) {
            MLLM_LOG_ERROR_LEGACY("Could not register Op Package: %s and interface provider: %s",
                                  opPackage[pathIdx].c_str(),
                                  opPackage[interfaceProviderIdx].c_str());
            return StatusCode::FAILURE;
        }
        MLLM_LOG_INFO_LEGACY("Registered Op Package: %s and interface provider: %s",
                             opPackage[pathIdx].c_str(),
                             opPackage[interfaceProviderIdx].c_str());
    }
    return StatusCode::SUCCESS;
}

// Create a Context in a backend.
StatusCode QNNBackend::createContext() {
    if (QNN_CONTEXT_NO_ERROR != m_qnnFunctionPointers.qnnInterface.contextCreate(m_backendHandle, m_deviceHandle, (const QnnContext_Config_t **)&m_contextConfig, &m_context)) {
        MLLM_LOG_ERROR_LEGACY("Could not create context");
        return StatusCode::FAILURE;
    }
    m_isContextCreated = true;
    return StatusCode::SUCCESS;
}

// Free context after done.
StatusCode QNNBackend::freeContext() {
    if (m_isContextCreated && QNN_CONTEXT_NO_ERROR != m_qnnFunctionPointers.qnnInterface.contextFree(m_context, m_profileBackendHandle)) {
        MLLM_LOG_ERROR_LEGACY("Could not free context");
        return StatusCode::FAILURE;
    }
    m_isContextCreated = false;
    return StatusCode::SUCCESS;
}

StatusCode QNNBackend::extractBackendProfilingInfo(
    Qnn_ProfileHandle_t profileHandle) {
    if (nullptr == m_profileBackendHandle) {
        MLLM_LOG_ERROR_LEGACY("Backend Profile handle is nullptr; may not be initialized.");
        return StatusCode::FAILURE;
    }
    const QnnProfile_EventId_t *profileEvents{nullptr};
    uint32_t numEvents{0};
    if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileGetEvents(profileHandle, &profileEvents, &numEvents)) {
        MLLM_LOG_ERROR_LEGACY("Failure in profile get events.");
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
    const QnnProfile_EventId_t *profileSubEvents{nullptr};
    uint32_t numSubEvents{0};
    if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileGetSubEvents(profileEventId, &profileSubEvents, &numSubEvents)) {
        MLLM_LOG_ERROR_LEGACY("Failure in profile get sub events.");
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
    if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileGetEventData(profileEventId, &eventData)) {
        MLLM_LOG_ERROR_LEGACY("Failure in profile get event type.");
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
            MLLM_LOG_WARN_LEGACY("Device property is not supported");
        }
        if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnStatus) {
            MLLM_LOG_ERROR_LEGACY("Device property is not known to backend");
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
            MLLM_LOG_ERROR_LEGACY("Failed to create device");
            return verifyFailReturnStatus(qnnStatus);
        }
    }
    return StatusCode::SUCCESS;
}

StatusCode QNNBackend::freeDevice() {
    if (nullptr != m_qnnFunctionPointers.qnnInterface.deviceFree) {
        auto qnnStatus = m_qnnFunctionPointers.qnnInterface.deviceFree(m_deviceHandle);
        if (QNN_SUCCESS != qnnStatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnStatus) {
            MLLM_LOG_ERROR_LEGACY("Failed to free device");
            return verifyFailReturnStatus(qnnStatus);
        }
    }
    return StatusCode::SUCCESS;
}

std::vector<Tensor> QNNBackend::runFunc(std::vector<std::string> out_names,
                                        TensorFuncType type,
                                        std::vector<float> float_args,
                                        std::vector<std::shared_ptr<Tensor>> input_tensors,
                                        bool in_place) {
    Module *module = input_tensors.empty() ? Module::llm_model_ptr : input_tensors[0]->module();
    assert(module != nullptr);
    auto &activation_tensors = module->activation_tensors;
    auto &activation_tensors_num = module->activation_tensors_num;

    std::vector<std::shared_ptr<Tensor>> output_ptrs;
    for (const auto &out_name : out_names) {
        if (activation_tensors.find(out_name) == activation_tensors.end()) {
            Backend *backend_h = Backend::global_backends[MLLM_CPU];
            if (!input_tensors.empty()) {
                backend_h = input_tensors[0]->backend();
            }
            activation_tensors[out_name] = std::make_shared<Tensor>(backend_h);
            activation_tensors[out_name]->setName(out_name);
            activation_tensors[out_name]->setModule(module);
            activation_tensors_num[out_name] = 0;
        }
        output_ptrs.push_back(activation_tensors[out_name]);
    }

    if (module->doLoad) {
        std::vector<Tensor> results;
        for (auto &out_tensor : output_ptrs) {
            results.push_back(*activation_tensors[out_tensor->name()]);
        }
        return results;
    }

    Backend *backend_h = Backend::global_backends[MLLM_CPU];
    if (!input_tensors.empty()) {
        backend_h = input_tensors[0]->backend();
    }
    TensorFunction *func = backend_h->funcCreate(type);

    std::vector<std::shared_ptr<Tensor>> input_ptrs;
    for (auto &tensor : input_tensors) {
        input_ptrs.push_back(activation_tensors[tensor->name()]);
    }
    // if (in_place) {
    //     for (size_t i = 0; i < input_tensors.size() && i < out_names.size(); ++i) {
    //         input_tensors[i]->setName(out_names[i]);
    //         output_ptrs.push_back(input_tensors[i]);
    //     }
    // }

#ifdef DEBUGOPTIME
    auto start_t = mllm_time_us();
#endif

    switch (Tensor::tensor_status) {
    case TENSOR_STATIC_INIT:
        func->reshape(output_ptrs, input_ptrs, float_args);
        func->setUp(output_ptrs, input_ptrs, float_args);
        break;
    case TENSOR_STATIC_READY:
        func->execute(output_ptrs, input_ptrs, float_args);
        break;
    case TENSOR_STATIC_TRACE:
        if (backend_h->type() == BackendType::MLLM_CPU) {
            Tracer::addTensorFunction(func, input_ptrs, output_ptrs, float_args);
        }
        break;
    default:
        break;
    }

    // if (Backend::global_backends.size() == 1) {
    //     for (auto input_tensor : input_ptrs) {
    //         auto it = activation_tensors_num.find(input_tensor->name());
    //         if (it != activation_tensors_num.end()) {
    //             switch (Tensor::tensor_status) {
    //             case TENSOR_STATIC_INIT:
    //                 it->second += 1;
    //                 break;
    //             case TENSOR_STATIC_READY:
    //                 it->second -= 1;
    //                 break;
    //             default:
    //                 break;
    //             }
    //             if (it->second == 0 && module_tensors[input_tensor->name()]->sequence() > 1 && module_tensors[input_tensor->name()]->ttype() != GRAPH_OUTPUT) {
    //                 activation_tensors[input_tensor->name()]->free();
    //             }
    //         }
    //     }
    // }

#ifdef DEBUGOPTIME
    if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        auto end_t = mllm_time_us();
        std::cout << (out_names.empty() ? "" : out_names[0]) << " | "
                  << Tensor::tensor_status << " time: "
                  << (end_t - start_t) / 1000.0F << "ms" << std::endl;
    }
#endif

#ifdef DEBUGSAVETENSOR
    for (auto &out_name : out_names) {
        activation_tensors[out_name]->saveNData<float>();
    }
#endif

    std::vector<Tensor> results;
    for (auto &out_tensor : output_ptrs) {
        results.emplace_back(*activation_tensors[out_tensor->name()]);
    }
    return results;
}
std::string name_num_to_X(const std::string &input_string) {
    std::regex pattern(R"(\.\d{1,3}\.)"); // Matches any number between 1 and 100 between two dots
    std::string replacement = ".X.";      // The string to replace the matched pattern with
    std::string output_string = std::regex_replace(input_string, pattern, replacement);
    return output_string;
}
std::string name_X_to_num(const std::string &input_string, int in_idx) {
    std::regex pattern(".X.");                                    // Matches any number between 1 and 100 between two dots
    std::string replacement = "." + std::to_string(in_idx) + "."; // The string to replace the matched pattern with
    std::string output_string = std::regex_replace(input_string, pattern, replacement);
    return output_string;
}
void init_reset_KVCache(string input_name, Module *module, int saved_list_idx, map<string, string> layername_2_tensorname, Backend *backend_) {
    map<string, shared_ptr<Tensor>> &activation_tensors = module->activation_tensors;
    vector<string> renameX_names;
    renameX_names.push_back(input_name);
    const vector<string> suffixs = {"-view", ".split-0", ".split-1", ".split-2", "-cat", "-split-0-48"};
    vector<string> new_names;
    bool can_break = true;
    auto in_x_name = renameX_names[0];
    while (can_break) {
        can_break = false;
        for (const auto &suffix : suffixs) {
            if (in_x_name.rfind(suffix) == (in_x_name.size() - suffix.size())) {
                const auto r_name = in_x_name.substr(0, in_x_name.size() - suffix.size());
                if (std::find(renameX_names.begin(), renameX_names.end(), r_name) == renameX_names.end() && std::find(new_names.begin(), new_names.end(), r_name) == new_names.end()) {
                    new_names.push_back(r_name);
                    in_x_name = r_name;
                    can_break = true;
                }
                break;
            }
        }
    }
    renameX_names.insert(renameX_names.end(), new_names.begin(), new_names.end());
    for (const auto x_name : renameX_names) {
        auto name = name_X_to_num(x_name, saved_list_idx);
        layername_2_tensorname[name] = name;
        activation_tensors[name] = std::make_shared<Tensor>(backend_);
        activation_tensors[name]->initFrom(*activation_tensors[x_name]);
        activation_tensors[name]->setName(name);
        activation_tensors[name]->setModule(module);
    }
}

std::vector<Tensor> QNNBackend::runLayer(Layer *layer, std::vector<Tensor> inputs, int N) {
    Module *module = inputs.empty() ? Module::llm_model_ptr : inputs[0].module();
    map<string, shared_ptr<Tensor>> &activation_tensors = module->activation_tensors;
    auto &activation_tensors_num = module->activation_tensors_num;
    // Module::runlistIdx = saved_list_idx;
    bool do_init = false;

    if (module->doLoad || !layer->inited_loaded) {
        // set backend to current module device and try to create op
        // use Module::tmp_device only when creating the op as the recersive module backend only handled in load and init stage
        layer->backend_ = Backend::global_backends[Module::tmp_device];
        do_init = !layer->inited_loaded;
        if (layer->op_ == nullptr) {
#ifdef USE_QNN
            if ((layer->param_["type"] == KVCACHE || layer->param_["type"] == KVCACHENPU) && (Backend::global_backends.find(MLLM_QNN) != Backend::global_backends.end())) {
                if (kv_cache_map.find(layer->name_) == kv_cache_map.end()) {
                    // for the prefill part, we need to create a new op
                    layer->param_["type"] = KVCACHENPU;
                    layer->op_ = layer->backend_->opCreate(layer->param_, layer->name_);
                    kv_cache_map[layer->name_] = layer->op_;
                } else {
#ifdef DEBUGPRINT
                    std::cout << name_ << " is shared used" << std::endl;
#endif
                    // for the decoding part, we need to get created op from global container
                    layer->op_ = kv_cache_map[layer->name_];
                }
            } else {
                layer->op_ = layer->backend_->opCreate(layer->param_, layer->name_);
            }
#else
            layer->op_ = layer->backend_->opCreate(layer->param_, layer->name_);
#endif
        }
        if (module->doLoad) {
            layer->op_->load(*module->loader);
            layer->inited_loaded = true;
        } else if (layer->loaded_param) {
            layer->inited_loaded = layer->loaded_param;
        } else {
            if (!layer->inited_loaded) {
                // module->loader = new ParamLoader("");
                // op_->load(*module->loader);
                auto empty_loader = new ParamLoader("");
                layer->op_->load(*empty_loader);
                layer->inited_loaded = true;
            }
        }
        vector<string> layer_next_names = {};
        if (N > 1) {
            for (int i = 0; i < N; ++i) {
                layer_next_names.push_back("out-" + layer->op_->name() + "-" + std::to_string(i));
            }
        } else {
            layer_next_names = {"out-" + layer->op_->name()};
        }
        for (const auto &layer_next_name : layer_next_names) {
            string next_name;
            if (Layer::use_layername_2_tensorname) {
                if (Layer::layername_2_tensorname.find(layer_next_name) == Layer::layername_2_tensorname.end()) {
                    if (layer->param_["type"] == KVCACHE) {
                        Layer::layername_2_tensorname[layer_next_name] = layer_next_name;
                        init_reset_KVCache(inputs[0].name(), module, layer->saved_list_idx, Layer::layername_2_tensorname, layer->backend_);
                    } else {
                        Layer::layername_2_tensorname[layer_next_name] = name_num_to_X(layer_next_name);
                    }
                }
                next_name = Layer::layername_2_tensorname[layer_next_name];
            } else {
                next_name = layer_next_name;
            }
            if (activation_tensors.find(next_name) == activation_tensors.end()) {
                activation_tensors[next_name] = std::make_shared<Tensor>(layer->backend_);
                activation_tensors[next_name]->setName(next_name);
                activation_tensors[next_name]->setModule(module);
                activation_tensors_num[next_name] = 0;
            }
        }
        if (module->doLoad) {
            vector<Tensor> output_result = {};
            for (const auto &layer_next_name : layer_next_names) {
                string next_name = Layer::use_layername_2_tensorname ? Layer::layername_2_tensorname[layer_next_name] : layer_next_name;
                output_result.push_back(*activation_tensors[next_name]);
            }
            return output_result;
        }
    }
    // input_tensors
    vector<shared_ptr<Tensor>> input_tensors;
    for (auto &input : inputs) {
        if (input.shouldInGraphs()) {
            auto input_name = input.name();
            if (layer->param_["type"] == KVCACHE && do_init && Layer::use_layername_2_tensorname) {
                input_name = name_X_to_num(input_name, layer->saved_list_idx);
            }
            input_tensors.push_back(activation_tensors[input_name]);
        } else {
            input_tensors.push_back(std::shared_ptr<Tensor>(&input, [](Tensor *) {}));
        }
    }
    // output_tensors
    vector<string> layer_next_names = {};
    if (N > 1) {
        for (int i = 0; i < N; ++i) {
            layer_next_names.push_back("out-" + layer->op_->name() + "-" + std::to_string(i));
        }
    } else {
        layer_next_names = {"out-" + layer->op_->name()};
    }
    vector<shared_ptr<Tensor>> output_tensors = {};
    for (const auto &layer_next_name : layer_next_names) {
        string next_name = Layer::use_layername_2_tensorname ? Layer::layername_2_tensorname[layer_next_name] : layer_next_name;
        output_tensors.push_back(activation_tensors[next_name]);
    }
#ifdef DEBUGOPTIME
    auto start_t = mllm_time_us();
#endif
    switch (Tensor::tensor_status) {
    case TENSOR_STATIC_INIT: {
        layer->op_->reshape(input_tensors, output_tensors);
        layer->op_->setUp(input_tensors, output_tensors);
        break;
    }
    case TENSOR_STATIC_READY: {
        layer->op_->execute(input_tensors, output_tensors);
        break;
    }
    case TENSOR_STATIC_TRACE: {
        if (layer->backend_->type() == BackendType::MLLM_CPU) {
            Tracer::addOp(layer->op_, input_tensors, output_tensors);
        }
        break;
    }
    default: {
        break;
    }
    }
    // if (Backend::global_backends.size() == 1) {
    //     for (auto input_tensor : input_tensors) {
    //         if ((activation_tensors_num.find(input_tensor->name()) != activation_tensors_num.end())) {
    //             switch (Tensor::tensor_status) {
    //             case TENSOR_STATIC_INIT: {
    //                 activation_tensors_num[input_tensor->name()] += 1;
    //                 break;
    //             }
    //             case TENSOR_STATIC_READY: {
    //                 activation_tensors_num[input_tensor->name()] -= 1;
    //                 break;
    //             }
    //             default: {
    //             }
    //             }
    //             if (activation_tensors_num[input_tensor->name()] == 0 && activation_tensors[input_tensor->name()]->sequence() > 1
    //                 && activation_tensors[input_tensor->name()]->ttype() != GRAPH_OUTPUT) {
    //                 activation_tensors[input_tensor->name()]->free();
    //                 // std::cout << input_tensor->name() << "|" << std::endl;
    //             }
    //         }
    //     }
    // }
#ifdef DEBUGOPTIME
    if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        auto end_t = mllm_time_us();
        std::cout << layer->op_->name() << " | " << Tensor::tensor_status << " time: " << (end_t - start_t) / 1000.0F << "ms" << std::endl;
    }
#endif
    vector<Tensor> output_result = {};
    for (const auto &layer_next_name : layer_next_names) {
        string next_name = Layer::use_layername_2_tensorname ? Layer::layername_2_tensorname[layer_next_name] : layer_next_name;
#ifdef DEBUGSAVETENSOR
        activation_tensors[next_name]->saveNData<float>(layer_next_name);
#endif
        output_result.push_back(*activation_tensors[next_name]);
    }
    return output_result;
}
std::vector<Tensor> QNNBackend::runForward(Module *module, std::vector<Tensor> inputs, std::vector<std::any> args) {
    // set static tmp_device to device_ to init layers' op
    auto previoud_device = Module::tmp_device;
    Module::tmp_device = module->device_;
    // Module Loading
    if (Module::llm_model_ptr && Module::llm_model_ptr->doLoad) {
        auto outputs = module->Forward(inputs, args);
        // for inner module, set output tensors to GRAPH_OUTPUT
        if (inputs[0].ttype() != TensorType::INPUT_TENSOR) { // XPUs' module should not be the outermost input tensor
            for (auto &output : outputs) {
                inputs[0].module()->activation_tensors[output.name()]->setTtype(GRAPH_OUTPUT);
            }
        }
        // set Module::tmp_device to previous device
        Module::tmp_device = previoud_device;
        return outputs;
    }
    // if (false) {
    //     inputs[0].setTtype(TensorType::INPUT_TENSOR);
    // }
    // Module setUp & execute
    if (inputs[0].ttype() == TensorType::INPUT_TENSOR) {
        if (module->prefilling_token_size_ == 0) { // first time init
            module->prefilling_token_size_ = inputs[0].sequence();
        } else if (module->decoding_token_size_ == 0) {
            module->decoding_token_size_ = inputs[0].sequence();
        }
        for (int i = 0; i < inputs.size(); i++) {
            auto &input = inputs[i];
            input.setName("input" + std::to_string(i));
            input.setTtype(TensorType::NORMAL_TENSOR);
            module->activation_tensors[input.name()] = std::shared_ptr<Tensor>(&input, [](Tensor *) {});
            module->activation_tensors[input.name()]->setName(input.name());
            module->activation_tensors[input.name()]->setModule(module);
        }
        Module::llm_model_ptr = module;
        Tensor::tensor_status = TENSOR_STATIC_INIT;

        uint64_t time_start = mllm_time_us();
        module->Forward(inputs, args);
        Tensor::tensor_status = TENSOR_STATIC_READY; // change to EAGER

        auto output = module->Forward(inputs, args);
        uint64_t time_end = mllm_time_us();

        double inference_time_ = (time_end - time_start) / 1000.0F; // ms
        module->inference_times_.push_back(inference_time_);

        Module::llm_model_ptr->op_transposed_flag = true;
        return output;
    } else { // inner Modules
        // offload according to the backends' info inited during loading
        if (Tensor::tensor_status == TENSOR_STATIC_INIT && module->device_ != MLLM_CPU) { // backend specific module reshape & setup
            if (Module::isMultiChunkPrefilling && !Module::isFirstChunk) {                // set to TENSOR_UNDEFINED and SKIP executing qnn layers
                Tensor::tensor_status = TENSOR_UNDEFINED;
                auto outputs = module->Forward(inputs, args);
                Tensor::tensor_status = TENSOR_STATIC_INIT;
                return outputs;
            }
            auto inputs_vec = vector<shared_ptr<Tensor>>();
            auto outputs_vec = vector<shared_ptr<Tensor>>();
            for (auto &i : inputs) {
                inputs_vec.push_back(inputs[0].module()->activation_tensors[i.name()]);
            }

            Backend::global_backends[module->device_]->onSetUpStart(inputs_vec, outputs_vec, module->getUniqueName());

            // for xnnpack currently
            for (auto &i : inputs) {
                i.uuid() = inputs[0].module()->activation_tensors[i.name()]->uuid();
            }

            auto outputs = module->Forward(inputs, args);
            for (auto &output : outputs) {
                outputs_vec.push_back(inputs[0].module()->activation_tensors[output.name()]);
            }
            Backend::global_backends[module->device_]->onSetUpEnd(inputs_vec, outputs_vec, module->getUniqueName());

            // for xnnpack currently
            for (auto &o : outputs) {
                o.uuid() = outputs[0].module()->activation_tensors[o.name()]->uuid();
            }

            return outputs;
        } else if (Tensor::tensor_status == TENSOR_STATIC_READY && module->device_ != MLLM_CPU) { // backend specific module execute
            auto inputs_vec = vector<shared_ptr<Tensor>>();
            auto outputs_vec = vector<shared_ptr<Tensor>>();
            for (auto &i : inputs) {
                inputs_vec.push_back(inputs[0].module()->activation_tensors[i.name()]);
            }

            auto outputs = module->Forward(inputs, args);

            for (auto &output : outputs) {
                outputs_vec.push_back(inputs[0].module()->activation_tensors[output.name()]);
            }
            Backend::global_backends[module->device_]->onExecuteStart(inputs_vec, outputs_vec, module->getUniqueName());

            Backend::global_backends[module->device_]->onExecuteEnd(outputs_vec, module->getUniqueName());

            // for xnnpack currently
            for (auto &o : outputs) {
                o.uuid() = outputs[0].module()->activation_tensors[o.name()]->uuid();
                o.forceResetHostPointer(outputs[0].module()->activation_tensors[o.name()]->rawHostPtr());
            }

            return outputs;
        } else if (Tensor::tensor_status == TENSOR_STATIC_TRACE && module->device_ != MLLM_CPU) {
            auto inputs_vec = vector<shared_ptr<Tensor>>();
            auto outputs_vec = vector<shared_ptr<Tensor>>();
            for (auto &i : inputs) {
                inputs_vec.push_back(inputs[0].module()->activation_tensors[i.name()]);
            }

            auto outputs = module->Forward(inputs, args);

            for (auto &output : outputs) {
                outputs_vec.push_back(inputs[0].module()->activation_tensors[output.name()]);
            }
            Tracer::addModule(inputs_vec, outputs_vec, module->getUniqueName());
            return outputs;
        }
        return module->Forward(inputs, args);
    }
}

} // namespace mllm