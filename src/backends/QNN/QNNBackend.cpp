#include <cstdint>
#include <inttypes.h>

#include <cstring>
#include <fstream>
#include <iostream>

#include "OpDefined.hpp"
#include "QNNBackend.hpp"
#include "QnnModel.hpp"
#include "Utils/BuildId.hpp"
#include "Utils/DataUtil.hpp"
#include "Utils/QnnSampleAppUtils.hpp"
#include "Utils/IOTensor.hpp"
#include "Utils/DynamicLoadUtil.hpp"
#include "Log/Logger.hpp"
#include "PAL/Directory.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"
#include "PAL/StringOp.hpp"
#include "PAL/DynamicLoading.hpp"
#include "PAL/GetOpt.hpp"
#include "WrapperUtils/QnnWrapperUtils.hpp"
#include "QNNMemoryManager.hpp"
#include "QnnTypes.h"
#include "QnnTypeMacros.hpp"

#include "Types.hpp"
#include "op/QNNAdd.hpp"
#include "op/QNNCausalMask.hpp"
#include "op/QNNGELU.hpp"
#include "op/QNNLinear.hpp"
#include "op/QNNLinear3D.hpp"
#include "op/QNNLinearFP.hpp"
#include "op/QNNLinearTest.hpp"
#include "op/QNNLinearINT8.hpp"
#include "op/QNNMatmul.hpp"
#include "op/QNNMatmulNT.hpp"
#include "op/QNNMatmulINT8.hpp"
#include "op/QNNMul.hpp"
#include "op/QNNLayerNorm.hpp"
#include "op/QNNRMSNorm.hpp"
#include "op/QNNRoPE.hpp"
#include "op/QNNScale.hpp"
#include "op/QNNSiLU.hpp"
#include "op/QNNSoftMax.hpp"
#include "op/QNNView.hpp"
#include "op/QNNKVCache.hpp"
#include "op/QNNWNop.hpp"

#include "op/QNNReLU.hpp"
#include "op/QNNQuantize.hpp"
#include "op/QNNDequantize.hpp"

#include "op/QNNMergeOutput.hpp"
#include "op/QNNSplitInput.hpp"

#define DEBUGPRINT
#ifdef DEBUGPRINT
#include "Timing.hpp"
#endif

using namespace qnn;
using namespace qnn::tools;
using namespace qnn::tools::sample_app;

// Flag to determine if Backend should node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

namespace mllm {

const std::string QNNBackend::s_defaultOutputPath = "./output";

void QNNBackend::registerOps() {
    addCreator(ADD, (QNNBackend::Creator *)new QNNAddCreator());
    addCreator(CAUSALMASK, (QNNBackend::Creator *)(new QNNCausalMaskCreator()));
    addCreator(MATMUL, (QNNBackend::Creator *)(new QNNMatmulCreator()));
    // addCreator(MATMUL, (QNNBackend::Creator *)(new QNNMatmulNTCreator()));
    addCreator(MATMULINT8, (QNNBackend::Creator *)(new QNNMatmulINT8Creator()));
    addCreator(RMSNORM, (QNNBackend::Creator *)(new QNNRMSNormCreator()));
    addCreator(LAYERNORM, (QNNBackend::Creator *)(new QNNLayerNormCreator()));
    addCreator(ROPE, (QNNBackend::Creator *)(new QNNRoPECreator()));
    addCreator(SCALE, (QNNBackend::Creator *)(new QNNScaleCreator()));
    addCreator(SILU, (QNNBackend::Creator *)(new QNNSiLUCreator()));
    addCreator(SOFTMAX, (QNNBackend::Creator *)(new QNNSoftMaxCreator()));
    addCreator(LINEAR, (QNNBackend::Creator *)(new QNNLinearCreator()));
    // addCreator(LINEAR, (QNNBackend::Creator *)(new QNNLinearFPCreator()));
    // addCreator(LINEAR, (QNNBackend::Creator *)(new QNNLinearTestCreator()));
    // addCreator(LINEAR, (QNNBackend::Creator *)(new QNNLinear3DCreator()));
    addCreator(LINEARINT8, (QNNBackend::Creator *)(new QNNLinearINT8Creator()));
    // addCreator(ATTENTION, (QNNBackend::Creator *)(new QNNAttentionCreator()));
    // addCreator(EMBEDDING, (QNNBackend::Creator *)(new QNNEmbeddingCreator()));
    addCreator(MUL, (QNNBackend::Creator *)(new QNNMulCreator()));
    addCreator(VIEW, (QNNBackend::Creator *)(new QNNViewCreator()));
    addCreator(KVCACHE, (QNNBackend::Creator *)(new QNNKVCacheCreator()));
    addCreator(WNOP, (QNNBackend::Creator *)(new QNNWNopCreator()));
    addCreator(RELU, (QNNBackend::Creator *)(new QNNReLUCreator()));
    addCreator(GELU, (QNNBackend::Creator *)(new QNNGELUCreator()));
    addCreator(QUANTIZE, (QNNBackend::Creator *)(new QNNQuantizeCreator()));
    addCreator(DEQUANTIZE, (QNNBackend::Creator *)(new QNNDequantizeCreator()));
    addCreator(MERGEOUTPUT, (QNNBackend::Creator *)(new QNNMergeOutputCreator()));
    addCreator(SPLITINPUT, (QNNBackend::Creator *)(new QNNSplitInputCreator()));
}

QNNBackend::QNNBackend(shared_ptr<MemoryManager> mm) :
    Backend(mm) {
    if (!log::initializeLogging()) {
        std::cerr << "ERROR: Unable to initialize logging!\n";
        return;
    }
    // TODO: make debug level configuable
    log::setLogLevel(QnnLog_Level_t::QNN_LOG_LEVEL_INFO);

    std::string backEndPath = "libQnnHtp.so";
    std::string opPackagePaths = "libQnnLLaMAPackage_CPU.so:LLaMAPackageInterfaceProvider:CPU,libQnnLLaMAPackage_HTP.so:LLaMAPackageInterfaceProvider:HTP";

    // TODO: make these configuable
    m_debug = false; // when set true, NATIVE tensor will be regared as APP_READ tensor
    m_outputDataType = iotensor::OutputDataType::FLOAT_AND_NATIVE;
    m_inputDataType = iotensor::InputDataType::NATIVE;
    m_profilingLevel = ProfilingLevel::BASIC;

    m_isBackendInitialized = false;
    m_isContextCreated = false;

    // config path strings
    split(m_opPackagePaths, opPackagePaths, ',');

    if (backEndPath.empty()) {
        std::exit(EXIT_FAILURE);
    }
    QNN_INFO("Backend: %s", backEndPath.c_str());

    // Load backend and validate all the required function symbols are resolved
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

    // init qnn resources
    {
        QNN_INFO("qnn-backend    build version: %s", getBuildId().c_str());
        QNN_INFO("Backend        build version: %s", getBackendBuildId().c_str());

        // initialize logging in the backend
        if (log::isLogInitialized()) {
            auto logCallback = log::getLogCallback();
            auto logLevel = log::getLogLevel();
            QNN_INFO("Initializing logging in the backend. Callback: [%p], Log Level: [%d]",
                     logCallback,
                     logLevel);
            if (QNN_SUCCESS != m_qnnFunctionPointers.qnnInterface.logCreate(logCallback, logLevel, &m_logHandle)) {
                QNN_WARN("Unable to initialize logging in the backend.");
            }
        } else {
            QNN_WARN("Logging not available in the backend.");
        }

        // initialize QnnBackend
        auto qnnStatus = m_qnnFunctionPointers.qnnInterface.backendCreate(
            m_logHandle, (const QnnBackend_Config_t **)m_backendConfig, &m_backendHandle);
        if (QNN_BACKEND_NO_ERROR != qnnStatus) {
            QNN_ERROR("Could not initialize backend due to error = %d", qnnStatus);
            this->reportError("Backend Initialization failure");
        }
        QNN_INFO("Initialize Backend Returned Status = %d", qnnStatus);
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

void QNNBackend::release() {
    // if (StatusCode::SUCCESS != this->freeContext()) {
    //     this->reportError("Context Free failure");
    // }

    auto devicePropertySupportStatus = this->isDevicePropertySupported();
    if (StatusCode::FAILURE != devicePropertySupportStatus) {
        auto freeDeviceStatus = this->freeDevice();
        if (StatusCode::SUCCESS != freeDeviceStatus) {
            this->reportError("Device Free failure");
        }
    }
}

void QNNBackend::onSetUpStart(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName) {
#ifdef DEBUGPRINT
    std::cout << "onSetUpStart" << std::endl;
#endif

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
    const QnnGraph_Config_t **graphConfigs = nullptr;
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

    // add input tensor to qnn
    // TODO: we believe it is NSHD now.
    uint32_t dimensionsInput[4] = {
                                static_cast<uint32_t>(inputs[0]->batch()),
                                static_cast<uint32_t>(inputs[0]->sequence()),
                                static_cast<uint32_t>(inputs[0]->head()),
                                static_cast<uint32_t>(inputs[0]->dimension()),
    };
    // for (int i = 0; i < 4; i++) {
    //     dimensionsInput[i] = inputs[0]->shape()[i];
    // }
    auto data_type = QNN_DATATYPE_FLOAT_32;
    if (inputs[0]->dtype() == MLLM_TYPE_I8) {
        std::cout << "QNN INT8 op" << std::endl;
        data_type = QNN_DATATYPE_UFIXED_POINT_8;
    }
    qnnModels_[qnnModelIndex_].addTensor(inputs[0]->name().c_str(), (Qnn_Tensor_t){
                                                                        .version = QNN_TENSOR_VERSION_1,
                                                                        {.v1 = {
                                                                             .id = 0,
                                                                             .name = inputs[0]->name().c_str(),
                                                                             .type = QNN_TENSOR_TYPE_APP_WRITE,
                                                                             .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                                                             .dataType = data_type,
                                                                             .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                                                                QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                                                                {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                                                             .rank = 4,
                                                                             .dimensions = dimensionsInput,
                                                                             .memType = QNN_TENSORMEMTYPE_RAW,
                                                                             {.clientBuf = {.data = nullptr,
                                                                                            .dataSize = 0}}}}});
    // create a new inputBuffer and outputBuffer for the graph
    inputBufferMap.insert(std::make_pair(graphName, std::vector<uint8_t *>(inputs.size())));
    outputBufferMap.insert(std::make_pair(graphName, std::vector<uint8_t *>(0)));
    
    currentInputBuffers = &inputBufferMap[graphName];
    currentOutputBuffers = &outputBufferMap[graphName];

    // push input tensors to the buffer list
    for(int i = 0; i < inputs.size(); i++) {
        (*currentInputBuffers)[i] = inputs[i]->hostPtr<uint8_t>();
    }
    
}

void QNNBackend::onSetUpEnd(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName) {
#ifdef DEBUGPRINT
    std::cout << "onSetUpEnd" << std::endl;
#endif
    // push output tensors to the buffer list
    currentOutputBuffers = &outputBufferMap[graphName];
    for (int i = 0; i < outputs.size(); i++) {
        currentOutputBuffers->push_back(outputs[i]->hostPtr<uint8_t>());
    }

    currentInputBuffers = &inputBufferMap[graphName];
    currentOutputBuffers = &outputBufferMap[graphName];
    qnnModelIndex_ = qnnModelIndexMap_[graphName];

    auto status = graphFinilize();
    if (qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR != status) {
        this->reportError("Graph Finalization failure");
    }

    auto returnStatus = StatusCode::SUCCESS;

    Qnn_Tensor_t *inputs_ = nullptr;
    Qnn_Tensor_t *outputs_ = nullptr;

    auto m_graphsInfo = m_graphsInfoMap_[qnnModelIndex_];

    for (size_t graphIdx = 0; graphIdx < 1; graphIdx++) {
        auto graphInfo = (*m_graphsInfo)[graphIdx];

        // QNN_DEBUG("input tensors: %d ", inputBuffers.size());
        // QNN_DEBUG("output tensors: %d ", outputBuffers.size());

        if (iotensor::StatusCode::SUCCESS != m_ioTensor.setupInputAndOutputTensors(&inputs_, &outputs_, (*m_graphsInfo)[graphIdx])) {
            QNN_ERROR("Error in setting up Input and output Tensors for graphIdx: %d", graphIdx);
            returnStatus = StatusCode::FAILURE;
            break;
        }

        // Todo only one graph now
        size_t totalCount = currentInputBuffers->size();
        if (iotensor::StatusCode::SUCCESS != m_ioTensor.populateInputTensors(graphIdx, *currentInputBuffers, inputs_, graphInfo, m_inputDataType)) {
            returnStatus = StatusCode::FAILURE;
        }

        // QNN_DEBUG("input tensors: %d ", (*m_graphsInfo)[graphIdx].numInputTensors);
        // QNN_DEBUG("output tensors: %d ", (*m_graphsInfo)[graphIdx].numOutputTensors);

        auto qnnMM = std::static_pointer_cast<QNNMemoryManager>(mem_manager_);

        // register input and output tensor to qnn shared buffers
        // TODO: currently must insure the inputs and outputs of mllm graph are the same as the qnn graph
        // op created io tensors (kvcache, wnop...) should be solved
        std::cout << "input tensors num:" << (*m_graphsInfo)[graphIdx].numInputTensors << std::endl;
        std::cout << "output tensors num:" << (*m_graphsInfo)[graphIdx].numOutputTensors << std::endl;

        std::cout << "input tensors num:" << currentInputBuffers->size() << std::endl;
        std::cout << "output tensors num:" << currentOutputBuffers->size() << std::endl;

        for (int i = 0; i < (*m_graphsInfo)[graphIdx].numInputTensors; i++) {
            // std::cout << "input name:" << inputs[i]->name() << std::endl;
            qnnMM->registerQnnTensor((*currentInputBuffers)[i], inputs_[i]);
            QNN_DEBUG("inputBuffers: %p ", (*currentInputBuffers)[i]);
        }
        for (int i = 0; i < (*m_graphsInfo)[graphIdx].numOutputTensors; i++) {
            // std::cout << "output name:" << outputs[i]->name() << std::endl;
            qnnMM->registerQnnTensor((*currentOutputBuffers)[i], outputs_[i]);
            QNN_DEBUG("outputBuffers: %p ", (*currentOutputBuffers)[i]);
        }
    }

    inputsMap_[qnnModelIndex_] = inputs_;
    outputsMap_[qnnModelIndex_] = outputs_;

}

void QNNBackend::onExecuteStart(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName) {
#ifdef DEBUGPRINT
    std::cout << "onExecuteStart" << std::endl;
#endif

    // to support multi-thread, we need local variable.
    // update currentInputBuffers, currentOutputBuffers, qnnModelIndex_
    auto t_qnnModelIndex_ = qnnModelIndexMap_[graphName];
    // qnn_wrapper_api::GraphInfo_t **m_graphsInfo = m_graphsInfoMap_[qnnModelIndex_];

    // std::cout << "graph name:" << graphName << std::endl;
    // std::cout << "output buffers size" << currentOutputBuffers->size() << std::endl;

// #ifdef QNN_ARM
    // reset the syncvar
    // for (auto t : syncVarTensors_) {
    //     t->setDataAt<uint32_t>(0, 0, 0, 0, 0);
    // }

    
// #endif
    qnn_wrapper_api::GraphInfo_t **m_graphsInfo = m_graphsInfoMap_[t_qnnModelIndex_];

    auto returnStatus = StatusCode::SUCCESS;

    for (size_t graphIdx = 0; graphIdx < 1; graphIdx++) {
        auto graphInfo = (*m_graphsInfo)[graphIdx];

        Qnn_Tensor_t *inputs_ = inputsMap_[t_qnnModelIndex_];
        Qnn_Tensor_t *outputs_ = outputsMap_[t_qnnModelIndex_];

        Qnn_ErrorHandle_t executeStatus = QNN_GRAPH_NO_ERROR;
        uint64_t t_start = mllm_time_us();
        executeStatus =
            m_qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                            inputs_,
                                                            graphInfo.numInputTensors,
                                                            outputs_,
                                                            graphInfo.numOutputTensors,
                                                            m_profileBackendHandle,
                                                            nullptr);
        // uint64_t t_end = mllm_time_us();
        // std::cout << "QNN execution time" << (t_end - t_start) / 1000.0F << " ms" << std::endl;

        // // print autoregressive latency.
        // FILE *fp = fopen("AR_latency.txt", "a");
    
        // // 检查文件是否成功打开
        // if (fp == NULL) {
        //     // 文件打开失败，输出错误消息并退出程序
        //     printf("无法打开文件或文件不存在。\n");
        // }
        
        // // 写入内容到文件
        // fprintf(fp, "QNN execution time %f ms\n", (t_end - t_start) / 1000.0F);
        
        // // 关闭文件
        // fclose(fp);

        if (QNN_GRAPH_NO_ERROR != executeStatus) {
            returnStatus = StatusCode::FAILURE;
        }

        // if (ProfilingLevel::OFF != m_profilingLevel) {
        //     extractBackendProfilingInfo(m_profileBackendHandle);
        // }
        // if (StatusCode::SUCCESS == returnStatus) {
        //     QNN_DEBUG("Successfully executed graphIdx: %d ", graphIdx);
        //     for (int oi = 0; oi < graphInfo.numOutputTensors; oi++) {
        //         auto output = outputs_[oi];
        //         // DEBUGLOG
        //         std::cout << "----------------" << std::endl;
        //         std::cout << "output name:" << output.v1.name << std::endl;
        //         // std::cout << "output id:" << output.v1.clientBuf.dataSize << std::endl;
        //         std::cout << "output type:" << output.v1.type << std::endl;
        //         std::cout << "output type:" << output.v1.dataType << std::endl;
        //     }
        // }

        // m_ioTensor.tearDownInputAndOutputTensors(
        //     inputs_, outputs_, graphInfo.numInputTensors, graphInfo.numOutputTensors);
        // inputs_ = nullptr;
        // outputs_ = nullptr;
        // if (StatusCode::SUCCESS != returnStatus) {
        //     std::cout << "tear down tensors fail" << std::endl;
        //     exit(-1);
        // }

        std::cout << "free graphs begin" << std::endl;
        qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
        m_graphsInfo = nullptr;
    }

}

void QNNBackend::onExecuteEnd() {
#ifdef QNN_ARM
    executeGraphsShared();
#else
    executeGraphs(inputBufferMap, outputBufferMap);
#endif
}

void QNNBackend::freeGraphDataStructure(string graphName) {

    auto it = qnnModelIndexMap_.find(graphName);
    if(it != qnnModelIndexMap_.end()) {
        std::cout << "free graph tensors begin" << std::endl;
        qnnModelIndex_ = it->second;

        qnnModels_[qnnModelIndex_].freeTensors();
        qnnModels_[qnnModelIndex_].clearGraph();
    }

    inputBufferMap[graphName].resize(0);
    outputBufferMap[graphName].resize(0);
}

void QNNBackend::afterAllGraphsExecute() {

    //TODO: dynamic free no useable graph.
    // clear old models.
    qnnModelIndexMap_.clear();

    auto qnnMM = std::static_pointer_cast<QNNMemoryManager>(mem_manager_);
    qnnMM->deRegisterQnnTensor();


    syncVarTensors_.resize(0);
    
    this->freeContext();

    // TODO: dynamic do not clear all the map.
    inputBufferMap.clear();
    outputBufferMap.clear();

    m_graphsInfoMap_.clear();
    inputsMap_.clear();
    outputsMap_.clear();
    
}

std::string QNNBackend::getBackendBuildId() {
    char *backendBuildId{nullptr};
    if (QNN_SUCCESS != m_qnnFunctionPointers.qnnInterface.backendGetBuildId((const char **)&backendBuildId)) {
        QNN_ERROR("Unable to get build Id from the backend.");
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
    if (params.size() > 0) {
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
    // Add all models to array to get graphsInfo
    qnn_wrapper_api::QnnModel *models[] = {&qnnModels_[qnnModelIndex_]};
    m_graphsCount = 1;
    // Populate the constructed graphs in provided output variables
    qnn_wrapper_api::ModelError_t err = qnn_wrapper_api::MODEL_NO_ERROR;
    qnn_wrapper_api::GraphInfo_t **m_graphsInfo = nullptr;
    
    VALIDATE(getGraphInfoFromModels(*models, m_graphsCount, &m_graphsInfo), err);
    // Graph finalize
    if (QNN_GRAPH_NO_ERROR != m_qnnFunctionPointers.qnnInterface.graphFinalize((*m_graphsInfo)[0].graph, m_profileBackendHandle, nullptr)) {
        return qnn_wrapper_api::ModelError_t::MODEL_GRAPH_ERROR;
    }
    if (ProfilingLevel::OFF != m_profilingLevel) {
        extractBackendProfilingInfo(m_profileBackendHandle);
    }
    m_graphsInfoMap_[qnnModelIndex_] = m_graphsInfo;

    return qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR;
}

qnn_wrapper_api::ModelError_t QNNBackend::modelAddTensor(std::string nodeName, Qnn_Tensor_t tensor) {
    return qnnModels_[qnnModelIndex_].addTensor(nodeName.c_str(), tensor);
}

StatusCode QNNBackend::initializeProfiling() {
    if (ProfilingLevel::OFF != m_profilingLevel) {
        QNN_INFO("Profiling turned on; level = %d", m_profilingLevel);
        if (ProfilingLevel::BASIC == m_profilingLevel) {
            QNN_INFO("Basic profiling requested. Creating Qnn Profile object.");
            if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileCreate(m_backendHandle, QNN_PROFILE_LEVEL_BASIC, &m_profileBackendHandle)) {
                QNN_WARN("Unable to create profile handle in the backend.");
                return StatusCode::FAILURE;
            }
        } else if (ProfilingLevel::DETAILED == m_profilingLevel) {
            QNN_INFO("Detailed profiling requested. Creating Qnn Profile object.");
            if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileCreate(m_backendHandle, QNN_PROFILE_LEVEL_DETAILED, &m_profileBackendHandle)) {
                QNN_ERROR("Unable to create profile handle in the backend.");
                return StatusCode::FAILURE;
            }
        }
    }
    return StatusCode::SUCCESS;
}

// Simple method to report error from app to lib.
void QNNBackend::reportError(const std::string &err) {
    QNN_ERROR("%s", err.c_str());
    exit(1);
}

// Terminate the backend after done.
StatusCode QNNBackend::terminateBackend() {
    if ((m_isBackendInitialized && nullptr != m_qnnFunctionPointers.qnnInterface.backendFree) && QNN_BACKEND_NO_ERROR != m_qnnFunctionPointers.qnnInterface.backendFree(m_backendHandle)) {
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
    const size_t pathIdx = 0;
    const size_t interfaceProviderIdx = 1;
    for (auto const &opPackagePath : m_opPackagePaths) {
        std::vector<std::string> opPackage;
        split(opPackage, opPackagePath, ':');
        QNN_DEBUG("opPackagePath: %s", opPackagePath.c_str());
        const char *target = nullptr;
        const size_t targetIdx = 2;
        if (opPackage.size() != 2 && opPackage.size() != 3) {
            QNN_ERROR("Malformed opPackageString provided: %s", opPackagePath.c_str());
            return StatusCode::FAILURE;
        }
        if (opPackage.size() == 3) {
            target = (char *)opPackage[targetIdx].c_str();
        }
        if (nullptr == m_qnnFunctionPointers.qnnInterface.backendRegisterOpPackage) {
            QNN_ERROR("backendRegisterOpPackageFnHandle is nullptr.");
            return StatusCode::FAILURE;
        }
        if (QNN_BACKEND_NO_ERROR != m_qnnFunctionPointers.qnnInterface.backendRegisterOpPackage(m_backendHandle, (char *)opPackage[pathIdx].c_str(), (char *)opPackage[interfaceProviderIdx].c_str(), target)) {
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
    if (QNN_CONTEXT_NO_ERROR != m_qnnFunctionPointers.qnnInterface.contextCreate(m_backendHandle, m_deviceHandle, (const QnnContext_Config_t **)&m_contextConfig, &m_context)) {
        QNN_ERROR("Could not create context");
        return StatusCode::FAILURE;
    }
    m_isContextCreated = true;
    return StatusCode::SUCCESS;
}

// Free context after done.
StatusCode QNNBackend::freeContext() {
    if (QNN_CONTEXT_NO_ERROR != m_qnnFunctionPointers.qnnInterface.contextFree(m_context, m_profileBackendHandle)) {
        QNN_ERROR("Could not free context");
        return StatusCode::FAILURE;
    }
    m_isContextCreated = false;
    return StatusCode::SUCCESS;
}

StatusCode QNNBackend::extractBackendProfilingInfo(
    Qnn_ProfileHandle_t profileHandle) {
    if (nullptr == m_profileBackendHandle) {
        QNN_ERROR("Backend Profile handle is nullptr; may not be initialized.");
        return StatusCode::FAILURE;
    }
    const QnnProfile_EventId_t *profileEvents{nullptr};
    uint32_t numEvents{0};
    if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileGetEvents(profileHandle, &profileEvents, &numEvents)) {
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
    const QnnProfile_EventId_t *profileSubEvents{nullptr};
    uint32_t numSubEvents{0};
    if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileGetSubEvents(profileEventId, &profileSubEvents, &numSubEvents)) {
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
    if (QNN_PROFILE_NO_ERROR != m_qnnFunctionPointers.qnnInterface.profileGetEventData(profileEventId, &eventData)) {
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

// executeGraphs() that load input/output buffers from CPU context
// inputBufferMap and outputBufferMap: graph_name -> graph input/output CPU buffers.
StatusCode QNNBackend::executeGraphs(std::map<std::string, std::vector<uint8_t *>> inputBufferMap, std::map<std::string, std::vector<uint8_t *>> outputBufferMap) {

    qnn_wrapper_api::GraphInfo_t **m_graphsInfo = m_graphsInfoMap_[qnnModelIndex_];

    auto returnStatus = StatusCode::SUCCESS;
    for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
        QNN_DEBUG("Starting execution for graphIdx: %d", graphIdx);
        if (graphIdx >= inputBufferMap.size()) {
            QNN_ERROR("No Inputs available for: %d", graphIdx);
            returnStatus = StatusCode::FAILURE;
            break;
        }

        Qnn_Tensor_t *inputs_ = inputsMap_[qnnModelIndex_];
        Qnn_Tensor_t *outputs_ = outputsMap_[qnnModelIndex_];

        auto graphInfo = (*m_graphsInfo)[graphIdx];
        if (!inputBufferMap.empty()) {

            size_t startIdx = 0;

            if (StatusCode::SUCCESS == returnStatus) {
                QNN_DEBUG("Successfully populated input tensors for graphIdx: %d", graphIdx);
                Qnn_ErrorHandle_t executeStatus = QNN_GRAPH_NO_ERROR;
                uint64_t t_start = mllm_time_us();

                executeStatus =
                    m_qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                                    inputs_,
                                                                    graphInfo.numInputTensors,
                                                                    outputs_,
                                                                    graphInfo.numOutputTensors,
                                                                    m_profileBackendHandle,
                                                                    nullptr);
                uint64_t t_end = mllm_time_us();
                std::cout << "QNN execution time" << (t_end - t_start) / 1000.0F << " ms" << std::endl;

                if (QNN_GRAPH_NO_ERROR != executeStatus) {
                    returnStatus = StatusCode::FAILURE;
                }
                if (StatusCode::SUCCESS == returnStatus) {
                    QNN_DEBUG("Successfully executed graphIdx: %d ", graphIdx);
                    for (int oi = 0; oi < graphInfo.numOutputTensors; oi++) {
                        auto output = outputs_[oi];
                        // DEBUGLOG
                        std::cout << "----------------" << std::endl;
                        std::cout << "output name:" << output.v1.name << std::endl;
                        std::cout << "output id:" << output.v1.clientBuf.dataSize << std::endl;
                        std::cout << "output type:" << output.v1.type << std::endl;
                        std::cout << "output type:" << output.v1.dataType << std::endl;
                        // m_ioTensor.writeOutputTensor(&output, outputBufferMap["graph"][oi]);
                        memcpy((*currentOutputBuffers)[oi], output.v1.clientBuf.data, output.v1.clientBuf.dataSize);
                    }
                }
            }
            if (StatusCode::SUCCESS != returnStatus) {
                QNN_ERROR("Execution of Graph: %d failed!", graphIdx);
                break;
            }
            if (ProfilingLevel::OFF != m_profilingLevel) {
                extractBackendProfilingInfo(m_profileBackendHandle);
            }
        }

        m_ioTensor.tearDownInputAndOutputTensors(
            inputs_, outputs_, graphInfo.numInputTensors, graphInfo.numOutputTensors);
        inputs_ = nullptr;
        outputs_ = nullptr;
        if (StatusCode::SUCCESS != returnStatus) {
            break;
        }
    }

    qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
    m_graphsInfo = nullptr;
    return returnStatus;
}

StatusCode QNNBackend::executeGraphsShared() {

    qnn_wrapper_api::GraphInfo_t **m_graphsInfo = m_graphsInfoMap_[qnnModelIndex_];

    auto returnStatus = StatusCode::SUCCESS;

    for (size_t graphIdx = 0; graphIdx < 1; graphIdx++) {
        auto graphInfo = (*m_graphsInfo)[graphIdx];

        Qnn_Tensor_t *inputs_ = inputsMap_[qnnModelIndex_];
        Qnn_Tensor_t *outputs_ = outputsMap_[qnnModelIndex_];

        Qnn_ErrorHandle_t executeStatus = QNN_GRAPH_NO_ERROR;
        uint64_t t_start = mllm_time_us();
        executeStatus =
            m_qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                            inputs_,
                                                            graphInfo.numInputTensors,
                                                            outputs_,
                                                            graphInfo.numOutputTensors,
                                                            m_profileBackendHandle,
                                                            nullptr);
        uint64_t t_end = mllm_time_us();
        std::cout << "QNN execution time" << (t_end - t_start) / 1000.0F << " ms" << std::endl;

        // print autoregressive latency.
        FILE *fp = fopen("AR_latency.txt", "a");
    
        // 检查文件是否成功打开
        if (fp == NULL) {
            // 文件打开失败，输出错误消息并退出程序
            printf("无法打开文件或文件不存在。\n");
        }
        
        // 写入内容到文件
        fprintf(fp, "QNN execution time %f ms\n", (t_end - t_start) / 1000.0F);
        
        // 关闭文件
        fclose(fp);

        if (QNN_GRAPH_NO_ERROR != executeStatus) {
            returnStatus = StatusCode::FAILURE;
        }
        if (StatusCode::SUCCESS == returnStatus) {
            QNN_DEBUG("Successfully executed graphIdx: %d ", graphIdx);
            for (int oi = 0; oi < graphInfo.numOutputTensors; oi++) {
                auto output = outputs_[oi];
                // DEBUGLOG
                std::cout << "----------------" << std::endl;
                std::cout << "output name:" << output.v1.name << std::endl;
                // std::cout << "output id:" << output.v1.clientBuf.dataSize << std::endl;
                std::cout << "output type:" << output.v1.type << std::endl;
                std::cout << "output type:" << output.v1.dataType << std::endl;
            }
        }

        // m_ioTensor.tearDownInputAndOutputTensors(
        //     inputs_, outputs_, graphInfo.numInputTensors, graphInfo.numOutputTensors);
        // inputs_ = nullptr;
        // outputs_ = nullptr;
        // if (StatusCode::SUCCESS != returnStatus) {
        //     std::cout << "tear down tensors fail" << std::endl;
        //     exit(-1);
        // }

        // std::cout << "free graphs begin" << std::endl;
        // qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
        // m_graphsInfo = nullptr;
    }
    return returnStatus;
}

StatusCode QNNBackend::executeGraphsSharedAutoregressive() {
    qnn_wrapper_api::GraphInfo_t **m_graphsInfo = m_graphsInfoMap_[qnnModelIndex_];

    auto returnStatus = StatusCode::SUCCESS;

    Qnn_Tensor_t *inputs_ = inputsMap_[qnnModelIndex_];
    Qnn_Tensor_t *outputs_ = outputsMap_[qnnModelIndex_];

    for (size_t graphIdx = 0; graphIdx < 1; graphIdx++) {
        auto graphInfo = (*m_graphsInfo)[graphIdx];

        Qnn_ErrorHandle_t executeStatus = QNN_GRAPH_NO_ERROR;
        uint64_t t_start = mllm_time_us();
        executeStatus =
            m_qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                            inputs_,
                                                            graphInfo.numInputTensors,
                                                            outputs_,
                                                            graphInfo.numOutputTensors,
                                                            m_profileBackendHandle,
                                                            nullptr);
        uint64_t t_end = mllm_time_us();
        std::cout << "QNN execution time" << (t_end - t_start) / 1000.0F << " ms" << std::endl;

        if (QNN_GRAPH_NO_ERROR != executeStatus) {
            returnStatus = StatusCode::FAILURE;
        }
        if (StatusCode::SUCCESS == returnStatus) {
            QNN_DEBUG("Successfully executed graphIdx: %d ", graphIdx);
            for (int oi = 0; oi < graphInfo.numOutputTensors; oi++) {
                auto output = outputs_[oi];
                // DEBUGLOG
                std::cout << "----------------" << std::endl;
                std::cout << "output name:" << output.v1.name << std::endl;
                // std::cout << "output id:" << output.v1.clientBuf.dataSize << std::endl;
                std::cout << "output type:" << output.v1.type << std::endl;
                std::cout << "output type:" << output.v1.dataType << std::endl;
            }
        }
    }
    return returnStatus;
}
} // namespace mllm