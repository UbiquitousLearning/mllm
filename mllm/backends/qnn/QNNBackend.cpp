#include <cstdint>

#include <cstring>
#include <iostream>
#include <memory>

#include "Backend.hpp"
#include "Context.hpp"
#include "Log.h"
#include "Module.hpp"
#include "Layer.hpp"
#include "OpDefined.hpp"
#include "QNNBackend.hpp"
#include "QNNUtils.hpp"
#include "QNNModel.hpp"
#include "QNNMemoryManager.hpp"
#include "QnnTypes.h"
#include "HTP/QnnHtpGraph.h"
#include "HTP/QnnHtpDevice.h"

#include "Types.hpp"
#include "op/QNNAdd.hpp"
#include "op/QNNCausalMask.hpp"
#include "op/QNNDequantizeAdd.hpp"
#include "op/QNNGELU.hpp"
#include "op/QNNQuickGELU.hpp"
#include "op/QNNLinearINT8.hpp"
#include "op/QNNMatmul.hpp"
#include "op/QNNMul.hpp"
#include "op/QNNLayerNorm.hpp"
#include "op/QNNRMSNorm.hpp"
#include "op/QNNRoPE.hpp"
#include "op/QNNRoPESimple.hpp"
#include "op/QNNScale.hpp"
#include "op/QNNSiLU.hpp"
#include "op/QNNSiLUHigh.hpp"
#include "op/QNNSoftMax.hpp"
#include "op/QNNSplit.hpp"
#include "op/QNNSubGraphFinalize.hpp"
#include "op/QNNSubGraphStart.hpp"
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
    addCreator(ROPESIMPLE, (QNNBackend::Creator *)(new QNNRoPESimpleCreator()));
    addCreator(IROPE, (QNNBackend::Creator *)(new QNNIRoPECreator()));
    addCreator(SCALE, (QNNBackend::Creator *)(new QNNScaleCreator()));
    addCreator(SILU, (QNNBackend::Creator *)(new QNNSiLUCreator()));
    addCreator(SILU_FULL_PRECISION, (QNNBackend::Creator *)(new QNNSiLUHighCreator()));
    addCreator(SOFTMAX, (QNNBackend::Creator *)(new QNNSoftMaxCreator()));
    addCreator(LINEAR, (QNNBackend::Creator *)(new QNNLinearINT8Creator()));
    addCreator(LINEARINT8, (QNNBackend::Creator *)(new QNNLinearINT8Creator()));
    addCreator(MUL, (QNNBackend::Creator *)(new QNNMulCreator()));
    addCreator(VIEW, (QNNBackend::Creator *)(new QNNViewCreator()));
    addCreator(RELU, (QNNBackend::Creator *)(new QNNReLUCreator()));
    addCreator(OP_GELU, (QNNBackend::Creator *)(new QNNGELUCreator()));
    addCreator(QUICKGLUE, (QNNBackend::Creator *)(new QNNQuickGELUCreator()));
    addCreator(QUANTIZE, (QNNBackend::Creator *)(new QNNQuantizeCreator()));
    addCreator(DEQUANTIZE, (QNNBackend::Creator *)(new QNNDequantizeCreator()));
    addCreator(DEQUANTIZEADD, (QNNBackend::Creator *)(new QNNDequantizeAddCreator()));
    addCreator(MERGEOUTPUT, (QNNBackend::Creator *)(new QNNMergeOutputCreator()));
    addCreator(SPLITINPUT, (QNNBackend::Creator *)(new QNNSplitInputCreator()));
    addCreator(TRANSPOSE, (QNNBackend::Creator *)(new QNNTransposeCreator()));
    addCreator(SUPERSILU, (QNNBackend::Creator *)(new QNNSuperSiLUCreator()));
    addCreator(SUBGRAPHSTART, (QNNBackend::Creator *)(new QNNSubGraphStartCreator()));
    addCreator(SUBGRAPHFINALIZE, (QNNBackend::Creator *)(new QNNSubGraphFinalizeCreator()));
    addCreator(SPLIT, (QNNBackend::Creator *)(new QNNSplitCreator()));
}

QNNBackend::QNNBackend(shared_ptr<MemoryManager> mm) :
    Backend(mm) {
    type_ = BackendType::MLLM_QNN; // used in Tensor.device()

    QnnLog_Level_t qnnLogLevel = QNN_LOG_LEVEL_WARN; // QNN_LOG_LEVEL_INFO; // QNN_LOG_LEVEL_WARN; // default QNN log level
    m_profilingLevel = ProfilingLevel::DETAILED;
    m_debug = false; // when set true, NATIVE tensor will be regared as APP_READ tensor

    loadQNNSymbol();
    loadQNNSystemSymbol();

    mRuntime = QNNRuntime::create(m_profilingLevel, qnnLogLevel);
    if (!mRuntime) {
        MLLM_LOG_ERROR_STREAM << "Failed to create QNN Runtime\n";
        exit(1);
    }

    // check QNN capability
    char *backendBuildId{nullptr};
    if (QNN_SUCCESS != mRuntime->qnnInterface.backendGetBuildId((const char **)&backendBuildId)) {
        MLLM_LOG_ERROR_LEGACY("Unable to get build Id from the backend.");
    }
    MLLM_LOG_INFO_STREAM << "QNN Backend Build Id: " << (backendBuildId == nullptr ? "" : backendBuildId);
    if (mRuntime->qnnInterface.propertyHasCapability(QNN_PROPERTY_TENSOR_SUPPORT_SPARSITY) == QNN_PROPERTY_SUPPORTED) {
        MLLM_LOG_INFO("QNN backend supports tensor sparsity");
    }
    if (mRuntime->qnnInterface.propertyHasCapability(QNN_PROPERTY_TENSOR_SUPPORT_DYNAMIC_DIMENSIONS) == QNN_PROPERTY_SUPPORTED) {
        MLLM_LOG_INFO("QNN backend supports dynamic dimensions");
    }
    if (mRuntime->qnnInterface.propertyHasCapability(QNN_PROPERTY_GRAPH_SUPPORT_EARLY_TERMINATION) == QNN_PROPERTY_SUPPORTED) {
        MLLM_LOG_INFO("QNN backend supports early termination");
    }

    // register ops
    this->registerOps();

    bool contextStatus = false;
    // check if the qnn_context.bin file exists
    if (!std::filesystem::exists("qnn_context.bin")) {
        contextStatus = mRuntime->createContext(m_context, nullptr);
    } else {
        contextStatus = mRuntime->retrieveContext(m_context, graphsInfo_, nullptr);
        // set the flag to indicate that the context is loaded from cache
        isFromCache = true;
        // fill qnnModelIndexMap_ info according to graphsInfo_
        for (size_t i = 0; i < graphsInfo_.size(); i++) {
            auto graphName = graphsInfo_[i]->graphName;
            qnnModelIndexMap_.insert(std::make_pair(graphName, i));
        }
    }
    if (!contextStatus) {
        MLLM_LOG_ERROR_STREAM << "Failed to create QNN context\n";
        exit(1);
    }

    // assign context to qnn memory manager
#ifdef QNN_ARM
    auto qnnMM = std::static_pointer_cast<QNNMemoryManager>(mem_manager_);
    qnnMM->setQnnInterfaceAndContext(mRuntime->qnnInterface, m_context);
#endif

    mPerf = QNNPerf::create(&mRuntime->qnnInterface);
    mPerf->setPowerConfigBurst();
    mPerf->setRpcLatencyAndPolling();
}

QNNBackend::~QNNBackend() {
    // free creaters in map_creator_
    for (auto &iter : map_creator_) {
        delete iter.second;
    }
    // free qnn backend resource
    mRuntime.release();
}

void QNNBackend::onSetUpStart(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName) {
    // if the graph already exists, just update the qnnModelIndex_ and set the input and output buffers
    if (qnnModelIndexMap_.find(graphName) != qnnModelIndexMap_.end()) {
        qnnModelIndex_ = qnnModelIndexMap_[graphName];

        inputBufferMap.insert(std::make_pair(graphName, std::vector<uint8_t *>(inputs.size())));
        outputBufferMap.insert(std::make_pair(graphName, std::vector<uint8_t *>()));

        currentInputBuffers = &inputBufferMap[graphName];
        currentOutputBuffers = &outputBufferMap[graphName];

        // push input tensors to the buffer list
        for (int i = 0; i < inputs.size(); i++) {
            (*currentInputBuffers)[i] = inputs[i]->hostPtr<uint8_t>();
        }
        return;
    }
    // else, create a QNNModel to build graph
    qnnModelIndex_ = qnnModels_.size();
    qnnModelIndexMap_.insert(std::make_pair(graphName, qnnModelIndex_));
    qnnModels_.push_back(QNNModel());

    // initialize qnn graph info, set graph info, graph count
    QnnHtpGraph_CustomConfig_t vtcmConfigInfo;
    vtcmConfigInfo.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE;
    vtcmConfigInfo.vtcmSizeInMB = 8;
    QnnGraph_Config_t vtcmConfig;
    vtcmConfig.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    vtcmConfig.customConfig = &vtcmConfigInfo;

    // QnnHtpGraph_CustomConfig_t htpThreadConfig;
    // htpThreadConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
    // htpThreadConfig.numHvxThreads = 6; // set a number. MAX = number of HVX HW blocks for that SoC
    // QnnGraph_Config_t threadConfig;
    // threadConfig.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    // threadConfig.customConfig = &htpThreadConfig;

    // supported in 2.34
    QnnHtpGraph_CustomConfig_t slcConfigInfo;
    slcConfigInfo.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
    slcConfigInfo.optimizationOption.type = QNN_HTP_GRAPH_OPTIMIZATION_TYPE_ENABLE_SLC_ALLOCATOR;
    slcConfigInfo.optimizationOption.floatValue = 1;
    QnnGraph_Config_t slcConfig;
    slcConfig.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    slcConfig.customConfig = &slcConfigInfo;

    const QnnGraph_Config_t *graphConfigList[] = {&vtcmConfig, &slcConfig, NULL};

    ModelError_t err = MODEL_NO_ERROR;
    if ((err = qnnModels_[qnnModelIndex_].initialize(mRuntime->backendHandle,
                                                     mRuntime->qnnInterface,
                                                     m_context,
                                                     graphName.c_str(),
                                                     m_debug,
                                                     DO_GRAPH_NODE_VALIDATIONS,
                                                     graphConfigList))
        != MODEL_NO_ERROR) {
        MLLM_LOG_ERROR_STREAM << "QNNBackend graph initialization failed for graph: " << graphName
                              << " with error code: " << static_cast<int>(err) << std::endl;
        exit(1);
    }

    for (auto &input : inputs) {
        Qnn_DataType_t data_type;
        auto quantizeDefined = QNN_DEFINITION_UNDEFINED;
        auto quantizeType = QNN_QUANTIZATION_ENCODING_UNDEFINED;
        float scale = 0.0f;
        switch (input->dtype()) {
        case MLLM_TYPE_F32:
            data_type = QNN_DATATYPE_FLOAT_32;
            break;
        case MLLM_TYPE_F16:
            data_type = QNN_DATATYPE_FLOAT_16;
            break;
        case MLLM_TYPE_I8: {
            data_type = QNN_DATATYPE_SFIXED_POINT_8;
            quantizeDefined = QNN_DEFINITION_DEFINED;
            quantizeType = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            scale = input->quant_param.scale;
            break;
        }
        case MLLM_TYPE_I16: {
            data_type = QNN_DATATYPE_SFIXED_POINT_16;
            quantizeDefined = QNN_DEFINITION_DEFINED;
            quantizeType = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
            scale = input->quant_param.scale;
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

bool QNNBackend::graphFinilize() {
    // Populate the constructed graphs in provided output variables
    GraphInfo_t *graphInfo = nullptr;

    // Graph finalize
    CALL_QNN(getSingleGraphInfoFromModel(qnnModels_[qnnModelIndex_], &graphInfo));
    if (QNN_GRAPH_NO_ERROR != mRuntime->qnnInterface.graphFinalize(graphInfo->graph, mRuntime->profileHandle, nullptr)) {
        return false;
    }
    CALL_QNN(qnnModels_[qnnModelIndex_].freeCachedTensors());
    if (ProfilingLevel::OFF != m_profilingLevel) {
        extractBackendProfilingInfo(mRuntime->profileHandle);
    }
    graphsInfo_.push_back(graphInfo);

    return true;
}

// finalize graph if needed, get qnn inputs and outputs tensors from graphInfo, register shared memory handles
void QNNBackend::onSetUpEnd(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName) {
    // online graph building, finalize graph
    if (!isFromCache) {
        PRINT_MEMORY_USAGE("before graph finilize")
        if (!graphFinilize()) {
            MLLM_LOG_ERROR("Graph Finalization failure");
            exit(1);
        }
        PRINT_MEMORY_USAGE("after graph finilize")
    }

    auto graphInfo = graphsInfo_[qnnModelIndex_];
    Qnn_Tensor_t *qnnInputs = graphInfo->inputTensors;
    Qnn_Tensor_t *qnnOutputs = graphInfo->outputTensors;

    auto qnnMM = std::static_pointer_cast<QNNMemoryManager>(mem_manager_);

    // register input and output tensor to qnn shared buffers
    // must insure the inputs and outputs of mllm graph are the same as the qnn graph
#ifdef DEBUGPRINT
    std::cout << "input tensors num:" << graphInfo->numInputTensors << std::endl;
    std::cout << "output tensors num:" << graphInfo->numOutputTensors << std::endl;
#endif

    for (int i = 0; i < graphInfo->numInputTensors; i++) {
        qnnInputs[i].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        qnnMM->registerQnnTensor((*currentInputBuffers)[i], qnnInputs[i]);
#ifdef DEBUGPRINT
        std::cout << "\nregistered input tensor backend staged ptr: " << (void *)(*currentInputBuffers)[i] << std::endl;
        std::cout << "qnn input tensor name: " << qnnInputs[i].v1.name << std::endl;
        std::cout << "qnn input tensor scale: " << qnnInputs[i].v1.quantizeParams.scaleOffsetEncoding.scale << std::endl;
#endif
    }
    for (int i = 0; i < graphInfo->numOutputTensors; i++) {
        qnnOutputs[i].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
        qnnMM->registerQnnTensor((*currentOutputBuffers)[i], qnnOutputs[i]);
#ifdef DEBUGPRINT
        std::cout << "\nregistered output tensor backend staged ptr: " << (void *)(*currentOutputBuffers)[i] << std::endl;
        std::cout << "qnn output tensor name: " << qnnOutputs[i].v1.name << std::endl;
        std::cout << "qnn output tensor scale: " << qnnOutputs[i].v1.quantizeParams.scaleOffsetEncoding.scale << std::endl;
#endif
    }
}

void QNNBackend::onExecuteStart(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName) {
    // to support multi-thread, we need local variable.
    // update currentInputBuffers, currentOutputBuffers, qnnModelIndex_
    auto t_qnnModelIndex_ = qnnModelIndexMap_[graphName];
    GraphInfo_t *graphInfo = graphsInfo_[t_qnnModelIndex_];

#ifdef DEBUGPRINT
    uint64_t t_start = mllm_time_us();
#endif
    if (mRuntime->qnnInterface.graphExecute(graphInfo->graph,
                                            graphInfo->inputTensors,
                                            graphInfo->numInputTensors,
                                            graphInfo->outputTensors,
                                            graphInfo->numOutputTensors,
                                            mRuntime->profileHandle,
                                            nullptr)
        != QNN_GRAPH_NO_ERROR) {
        MLLM_LOG_ERROR_STREAM << "Error in executing graph: " << graphName << std::endl;
    }
#ifdef DEBUGPRINT
    uint64_t t_end = mllm_time_us();
    std::cout << "QNN execution time " << (t_end - t_start) / 1000.0F << " ms" << std::endl;
#endif

    if (ProfilingLevel::OFF != m_profilingLevel) {
        extractBackendProfilingInfo(mRuntime->profileHandle);
    }
}

void QNNBackend::graphAddNode(string name,
                              string nodeType,
                              std::vector<string> inputTensorNames,
                              std::vector<Qnn_Tensor_t> outputTensors,
                              std::vector<Qnn_Param_t> params,
                              string packageName) {
    // graph has been built
    if (isFromCache) {
        return;
    }
    CALL_QNN(qnnModels_[qnnModelIndex_].addNode(
        QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
        name.c_str(),           // Node Name
        packageName.c_str(),    // Package Name
        nodeType.c_str(),       // Qnn Node Type
        params,                 // Node Params
        inputTensorNames,       // Input Tensor Names
        outputTensors           // Output Tensors
        ));
}

void QNNBackend::modelAddTensor(std::string nodeName, Qnn_Tensor_t tensor) {
    // graph has been built
    if (isFromCache) {
        return;
    }
    // std::cout << "nodeName" << nodeName << std::endl;
    CALL_QNN(qnnModels_[qnnModelIndex_].addTensor(nodeName.c_str(), tensor));
}

void QNNBackend::extractBackendProfilingInfo(
    Qnn_ProfileHandle_t profileHandle) {
    if (nullptr == mRuntime->profileHandle) {
        MLLM_LOG_ERROR("Backend Profile handle is nullptr; may not be initialized.");
        return;
    }
    const QnnProfile_EventId_t *profileEvents{nullptr};
    uint32_t numEvents{0};
    if (QNN_PROFILE_NO_ERROR != mRuntime->qnnInterface.profileGetEvents(profileHandle, &profileEvents, &numEvents)) {
        MLLM_LOG_ERROR("Failure in profile get events.");
        return;
    }

    MLLM_LOG_INFO_STREAM << "Profile Events: [" << profileEvents << "], numEvents: " << numEvents << std::endl;
    for (size_t event = 0; event < numEvents; event++) {
        extractProfilingEvent(*(profileEvents + event));
        extractProfilingSubEvents(*(profileEvents + event));
    }
}

void QNNBackend::extractProfilingSubEvents(
    QnnProfile_EventId_t profileEventId) {
    const QnnProfile_EventId_t *profileSubEvents{nullptr};
    uint32_t numSubEvents{0};
    if (QNN_PROFILE_NO_ERROR != mRuntime->qnnInterface.profileGetSubEvents(profileEventId, &profileSubEvents, &numSubEvents)) {
        MLLM_LOG_ERROR_LEGACY("Failure in profile get sub events.");
        return;
    }
    MLLM_LOG_INFO_STREAM << "ProfileSubEvents: [" << profileSubEvents << "], numSubEvents: " << numSubEvents << std::endl;
    for (size_t subEvent = 0; subEvent < numSubEvents; subEvent++) {
        extractProfilingEvent(*(profileSubEvents + subEvent));
        extractProfilingSubEvents(*(profileSubEvents + subEvent));
    }
}

void QNNBackend::extractProfilingEvent(
    QnnProfile_EventId_t profileEventId) {
    QnnProfile_EventData_t eventData;
    if (QNN_PROFILE_NO_ERROR != mRuntime->qnnInterface.profileGetEventData(profileEventId, &eventData)) {
        MLLM_LOG_ERROR_LEGACY("Failure in profile get event type.");
        return;
    }
    MLLM_LOG_INFO_STREAM << "Printing Event Info - Event Type: [" << eventData.type
                         << "], Event Value: [" << eventData.value
                         << "], Event Identifier: [" << eventData.identifier
                         << "], Event Unit: [" << eventData.unit << "]" << std::endl;
}

void QNNBackend::saveQNNContext() {
    uint64_t binarySize, writtenSize;

    mRuntime->qnnInterface.contextGetBinarySize(m_context, &binarySize);

    std::unique_ptr<uint8_t[]> binaryBuffer(new uint8_t[binarySize]);

    mRuntime->qnnInterface.contextGetBinary(m_context, reinterpret_cast<void *>(binaryBuffer.get()), binarySize, &writtenSize);

    if (binarySize < writtenSize) {
        MLLM_LOG_ERROR_STREAM << "QNN context binary size mismatch: expected " << binarySize
                              << " bytes, but wrote " << writtenSize << " bytes." << std::endl;
    }
    std::ofstream file("qnn_context.bin", std::ios::binary);
    file.write(reinterpret_cast<char *>(binaryBuffer.get()), writtenSize);
    file.close();

    std::cout << "QNN context saved to qnn_context.bin written " << writtenSize << std::endl;
}
std::vector<Tensor> QNNBackend::runOp(Op *op, std::vector<Tensor> inputs, std::vector<std::string> out_names, bool in_place) {
    Module *module = inputs.empty() ? Module::llm_model_ptr : inputs[0].module();
    assert(module != nullptr);
    auto &activation_tensors = module->activation_tensors;
    auto &activation_tensors_num = module->activation_tensors_num;

    std::vector<std::shared_ptr<Tensor>> output_ptrs;
    for (const auto &out_name : out_names) {
        if (activation_tensors.find(out_name) == activation_tensors.end()) {
            Backend *backend_h = Backend::global_backends[MLLM_CPU].get();
            if (!inputs.empty()) {
                backend_h = inputs[0].backend();
            }
            activation_tensors[out_name] = std::make_shared<Tensor>(backend_h);
            activation_tensors[out_name]->setName(out_name);
            activation_tensors[out_name]->setModule(module);
            activation_tensors_num[out_name] = 0;
        }
        output_ptrs.push_back(activation_tensors[out_name]);
    }
    Backend *backend_h = Backend::global_backends[MLLM_CPU].get();
    if (!inputs.empty()) {
        backend_h = inputs[0].backend();
    }
    if (module->doLoad) {
        std::vector<Tensor> results;
        for (auto &out_tensor : output_ptrs) {
            results.push_back(*activation_tensors[out_tensor->name()]);
        }
        return results;
    }

    std::vector<std::shared_ptr<Tensor>> input_ptrs;
    for (auto &tensor : inputs) {
        input_ptrs.push_back(activation_tensors[tensor.name()]);
    }

#ifdef DEBUGOPTIME
    auto start_t = mllm_time_us();
#endif

    switch (Tensor::tensor_status) {
    case TENSOR_STATIC_INIT:
        op->reshape(input_ptrs, output_ptrs);
        op->setUp(input_ptrs, output_ptrs);
        break;
    case TENSOR_STATIC_READY:
        op->execute(input_ptrs, output_ptrs);
        break;
    case TENSOR_STATIC_TRACE:
        if (backend_h->type() == BackendType::MLLM_CPU) {
            Tracer::addOp(op, input_ptrs, output_ptrs);
        } else if (op->type() == SUBGRAPHSTART) { // begin of QNN graph
            Tracer::addModule(input_ptrs, {}, op->name());
        }
        break;
        break;
    default:
        break;
    }
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

/*
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
            Backend *backend_h = Context::Instance().globalBackends(MLLM_CPU);
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

    Backend *backend_h = Context::Instance().globalBackends(MLLM_CPU);
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
*/
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
        // layer->backend_ = Context::Instance().globalBackends(Module::tmp_device);
        layer->backend_ = Backend::global_backends[Module::tmp_device].get();
        do_init = !layer->inited_loaded;
        if (layer->op_ == nullptr) {
            // std::cout << "asdsa  " << layer->name_ << std::endl;
            if (layer->param_["type"] == KVCACHE || layer->param_["type"] == KVCACHENPU) {
                // std::cout << layer->name_ << std::endl;
                if (kv_cache_map.find(layer->name_) == kv_cache_map.end()) {
                    // std::cout << layer->name_ << " is first used" << std::endl;
                    // for the prefill part, we need to create a new op
                    layer->param_["type"] = KVCACHENPU;
                    layer->op_ = layer->backend_->opCreate(layer->param_, layer->name_);
                    kv_cache_map[layer->name_] = layer->op_;
                } else {
                    // #ifdef DEBUGPRINT
                    // std::cout << layer->name_ << " is shared used" << std::endl;
                    // #endif
                    // for the decoding part, we need to get created op from global container
                    layer->op_ = kv_cache_map[layer->name_];
                }
            } else {
                layer->op_ = layer->backend_->opCreate(layer->param_, layer->name_);
            }
        }
        if (layer->param_["type"] == SUBGRAPHFINALIZE) {
            for (auto &input : inputs) {
                activation_tensors[input.name()]->setTtype(GRAPH_OUTPUT);
            }
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
            // NOTE: QNN is using CPU ViT
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
            } else if (Context::Instance().inference_state().getIsCPUViT() && layer_next_name.find("visual") != string::npos) {
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
                string next_name;
                // NOTE: QNN is using CPU ViT
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
                } else if (Context::Instance().inference_state().getIsCPUViT() && layer_next_name.find("visual") != string::npos) {
                    next_name = Layer::layername_2_tensorname[layer_next_name];
                } else {
                    next_name = layer_next_name;
                }
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
        string next_name;
        // NOTE: QNN is using CPU ViT
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
        } else if (Context::Instance().inference_state().getIsCPUViT() && layer_next_name.find("visual") != string::npos) {
            next_name = Layer::layername_2_tensorname[layer_next_name];
        } else {
            next_name = layer_next_name;
        }
        output_tensors.push_back(activation_tensors[next_name]);
    }
#ifdef DEBUGOPTIME
    auto start_t = mllm_time_us();
#endif
    switch (Tensor::tensor_status) {
    case TENSOR_STATIC_INIT: {
        if (Context::Instance().inference_state().isQnnGraphFrozen() && layer->backend_->type() == MLLM_QNN) {
            break;
        }
        // std::cout << "================={Layer: " << std::endl;
        // std::cout << layer->op_->name() << std::endl;
        // for (const auto &in_tensor : input_tensors) {
        //     std::cout << "    in tensor: " << in_tensor->name() << " dtype=" << in_tensor->dtype() << " " << in_tensor->batch() << ",  " << in_tensor->head() << ", " << in_tensor->sequence() << ",  " << in_tensor->dimension() << "   ctype " << in_tensor->ctype() << "   dtype " << in_tensor->dtype() << std::endl;
        // }
        layer->op_->reshape(input_tensors, output_tensors);
        layer->op_->setUp(input_tensors, output_tensors);
        // for (const auto &in_tensor : output_tensors) {
        //     std::cout << "    ot tensor: " << in_tensor->name() << " dtype=" << in_tensor->dtype() << " " << in_tensor->batch() << ",  " << in_tensor->head() << ", " << in_tensor->sequence() << ",  " << in_tensor->dimension() << "   ctype " << in_tensor->ctype() << "   dtype " << in_tensor->dtype() << std::endl;
        // }
        // std::cout << "=================Layer}: " << std::endl;
        break;
    }
    case TENSOR_STATIC_READY: {
        if (Context::Instance().inference_state().isQnnGraphFrozen() && layer->backend_->type() == MLLM_QNN && layer->param_["type"] != SUBGRAPHSTART) {
            break;
        }
        layer->op_->execute(input_tensors, output_tensors);
        break;
    }
    case TENSOR_STATIC_TRACE: {
        if (layer->backend_->type() == BackendType::MLLM_CPU) {
            Tracer::addOp(layer->op_, input_tensors, output_tensors);
        } else if (layer->param_["type"] == SUBGRAPHSTART) { // begin of QNN graph
            Tracer::addModule(input_tensors, {}, layer->op_->name());
        }
        break;
    }
    default: {
        break;
    }
    }

#ifdef DEBUGOPTIME
    if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        auto end_t = mllm_time_us();
        std::cout << layer->op_->name() << " | " << Tensor::tensor_status << " time: " << (end_t - start_t) / 1000.0F << "ms" << std::endl;
    }
#endif
    vector<Tensor> output_result = {};
    for (const auto &layer_next_name : layer_next_names) {
        string next_name;
        // NOTE: QNN is using CPU ViT
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
        } else if (Context::Instance().inference_state().getIsCPUViT() && layer_next_name.find("visual") != string::npos) {
            next_name = Layer::layername_2_tensorname[layer_next_name];
        } else {
            next_name = layer_next_name;
        }
#ifdef DEBUGSAVETENSOR
        activation_tensors[next_name]->saveNData<float>(layer_next_name);
#endif
        output_result.push_back(*activation_tensors[next_name]);
    }
    return output_result;
}
std::vector<Tensor> QNNBackend::runForward(Module *module, std::vector<Tensor> inputs, std::vector<std::any> args) {
    // Module Loading
    if (Module::llm_model_ptr && Module::llm_model_ptr->doLoad) {
        auto outputs = module->Forward(inputs, args);
        return outputs;
    }

    // Module setUp & execute
    if (inputs[0].ttype() == TensorType::INPUT_TENSOR) {
        if (module->prefilling_token_size_ == 0) { // first time init
            module->prefilling_token_size_ = inputs[0].sequence() * inputs[0].batch();
        } else if (module->decoding_token_size_ == 0) {
            module->decoding_token_size_ = inputs[0].sequence() * inputs[0].batch();
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
        return module->Forward(inputs, args);
    }
}

QNNPerf::QNNPerf(const QNN_INTERFACE_VER_TYPE *qnnInterface) {
    assert(qnnInterface != nullptr);
    mQnnInterface = qnnInterface;

    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    CALL_QNN(mQnnInterface->deviceGetInfrastructure(&deviceInfra));
    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    mPerfInfra = htpInfra->perfInfra;

    uint32_t deviceId = 0;
    uint32_t coreId = 0;
    CALL_QNN(mPerfInfra.createPowerConfigId(deviceId, coreId, &mPowerConfigId));

    mPowerConfigBurst = {
        .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
        .dcvsV3Config = {
            .contextId = mPowerConfigId, // use the power config id created
            .setDcvsEnable = 1,
            .dcvsEnable = 0, // 1- To enable Dcvs and consider dcvs power mode, 0- To disable dcvs
            .powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,
            .setSleepLatency = 1, // True to consider Latency parameter otherwise False
            .sleepLatency = 40,   // set dsp sleep latency ranges 10-65535 micro sec, refer hexagon sdk
            .setSleepDisable = 1, // True to consider sleep disable/enable parameter otherwise False
            .sleepDisable = 1,    // True to disable sleep, False to re-enable sleep
            .setBusParams = 1,    // True to consider Bus parameter otherwise False
            .busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
            .busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
            .busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
            .setCoreParams = 1, // True to consider Core parameter otherwise False
            .coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
            .coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
            .coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
        },
    };

    mPowerConfigBalanced = {
        .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
        .dcvsV3Config = {
            .contextId = mPowerConfigId, // use the power config id created
            .setDcvsEnable = 1,
            .dcvsEnable = 1, // 1- To enable Dcvs and consider dcvs power mode, 0- To disable dcvs
            .powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_ADJUST_UP_DOWN,
            .setSleepLatency = 1, // True to consider Latency parameter otherwise False
            .sleepLatency = 1000, // set dsp sleep latency ranges 10-65535 micro sec, refer hexagon sdk
            .setSleepDisable = 1, // True to consider sleep disable/enable parameter otherwise False
            .sleepDisable = 0,    // True to disable sleep, False to re-enable sleep
            .setBusParams = 1,    // True to consider Bus parameter otherwise False
            .busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO,
            .busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO,
            .busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO,
            .setCoreParams = 1, // True to consider Core parameter otherwise False
            .coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO,
            .coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO,
            .coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO,
        },
    };
}

// destory power config
QNNPerf::~QNNPerf() {
    CALL_QNN(mPerfInfra.destroyPowerConfigId(mPowerConfigId));
}

void QNNPerf::setRpcLatencyAndPolling() {
    // set RPC Control Latency
    QnnHtpPerfInfrastructure_PowerConfig_t rpcControlLatency; // refer QnnHtpPerfInfrastructure.h
    ::memset(&rpcControlLatency, 0, sizeof(rpcControlLatency));
    rpcControlLatency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpcControlLatency.rpcControlLatencyConfig = 100; // use rpc control latency recommended 100 us, refer hexagon sdk
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs1[] = {&rpcControlLatency, NULL};

    CALL_QNN(mPerfInfra.setPowerConfig(mPowerConfigId, powerConfigs1)); // set RPC latency config on power config ID created

    // set RPC Polling
    QnnHtpPerfInfrastructure_PowerConfig_t rpcPollingTime; // refer QnnHtpPerfInfrastructure.h
    ::memset(&rpcPollingTime, 0, sizeof(rpcPollingTime));
    rpcPollingTime.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpcPollingTime.rpcPollingTimeConfig = 9999; // use rpc polling time recommended 0-10000 us
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs2[] = {&rpcPollingTime, NULL};

    CALL_QNN(mPerfInfra.setPowerConfig(mPowerConfigId, powerConfigs2)); // set RPC polling config on power config ID created
}

void QNNPerf::setPowerConfigBurst() {
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs[] = {&mPowerConfigBurst, NULL};
    CALL_QNN(mPerfInfra.setPowerConfig(mPowerConfigId, powerConfigs));
}

void QNNPerf::setPowerConfigBalanced() {
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs[] = {&mPowerConfigBalanced, NULL};
    CALL_QNN(mPerfInfra.setPowerConfig(mPowerConfigId, powerConfigs));
}

QNNRuntime::~QNNRuntime() {
    // Free Profile
    if (profileHandle != nullptr) {
        CALL_QNN(qnnInterface.profileFree(profileHandle));
    }

    // Free Device
    CALL_QNN(qnnInterface.deviceFree(deviceHandle));

    // Free Backend
    CALL_QNN(qnnInterface.backendFree(backendHandle));

    // Free Log
    CALL_QNN(qnnInterface.logFree(logHandle));
}

void __mllmLoggerCallback4QnnLogger(const char *fmt, QnnLog_Level_t level, uint64_t times_tamp,
                                    va_list argp) {
    const char *level_str = "";
    switch (level) {
    case QNN_LOG_LEVEL_ERROR: level_str = "[ERROR]"; break;
    case QNN_LOG_LEVEL_WARN: level_str = "[WARN]"; break;
    case QNN_LOG_LEVEL_INFO: level_str = "[INFO]"; break;
    case QNN_LOG_LEVEL_DEBUG: level_str = "[DEBUG]"; break;
    case QNN_LOG_LEVEL_VERBOSE: level_str = "[VERBOSE]"; break;
    case QNN_LOG_LEVEL_MAX: level_str = "[UNKNOWN]"; break;
    }

    double ms = (double)times_tamp / 1000000.0;

    {
        fprintf(stdout, "QnnLogger(%8.1fms, %ld) %s: ", ms, times_tamp, level_str);
        vfprintf(stdout, fmt, argp);
    }
}

QNNRuntime *QNNRuntime::initRuntime(ProfilingLevel profilingLevel, QnnLog_Level_t qnnLogLevel) {
    // Create Interface
    QNN_INTERFACE_VER_TYPE qnnInterface{};
    {
        QnnInterface_t **interfaceProviders = nullptr;
        uint32_t numProviders = 0;
        if (QnnInterface_getProviders((const QnnInterface_t ***)&interfaceProviders, &numProviders) != QNN_SUCCESS) {
            MLLM_LOG_ERROR_STREAM << "Failed to call 'QnnInterface_getProviders'." << std::endl;
            return nullptr;
        }
        if (interfaceProviders == nullptr) {
            MLLM_LOG_ERROR_STREAM << "Failed to get interface providers: null interface providers received." << std::endl;
            return nullptr;
        }
        if (numProviders == 0) {
            MLLM_LOG_ERROR_STREAM << "Failed to get interface providers: 0 interface providers." << std::endl;
            return nullptr;
        }
        bool foundValidInterface = false;
        for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
            if (QNN_API_VERSION_MAJOR == interfaceProviders[pIdx]->apiVersion.coreApiVersion.major && QNN_API_VERSION_MINOR <= interfaceProviders[pIdx]->apiVersion.coreApiVersion.minor) {
                foundValidInterface = true;
                qnnInterface = interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
                break;
            }
        }
        if (!foundValidInterface) {
            MLLM_LOG_ERROR_STREAM << "Failed to find a valid QNN interface provider." << std::endl;
            return nullptr;
        }
    }

    // Create Log
    Qnn_LogHandle_t logHandle = nullptr;
    {
        QnnLog_Callback_t logCallback = __mllmLoggerCallback4QnnLogger;

        if ((QNN_GET_ERROR_CODE(qnnInterface.logCreate(logCallback, QNN_LOG_LEVEL_ERROR, &logHandle)) != QNN_SUCCESS) || (logHandle == nullptr)) {
            MLLM_LOG_ERROR_STREAM << "Failed to initialize logging in the backend." << std::endl;
            return nullptr;
        }
    }

    // Create Backend
    Qnn_BackendHandle_t backendHandle = nullptr;
    {
        const QnnBackend_Config_t **backendConfig = nullptr;
        if ((QNN_GET_ERROR_CODE(qnnInterface.backendCreate(logHandle, backendConfig, &backendHandle)) != QNN_SUCCESS) || (backendHandle == nullptr)) {
            MLLM_LOG_ERROR_STREAM << "Failed to create the backend." << std::endl;
            return nullptr;
        }
    }

    // Create Device
    Qnn_DeviceHandle_t deviceHandle = nullptr;
    {
        // Check whether the device API is supported.
        if (nullptr != qnnInterface.propertyHasCapability) {
            auto qnnStatus =
                qnnInterface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
            if (QNN_PROPERTY_NOT_SUPPORTED == qnnStatus) {
                MLLM_LOG_WARN_LEGACY("Device property is not supported");
                return nullptr;
            }
            if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnStatus) {
                MLLM_LOG_ERROR_LEGACY("Device property is not known to backend");
                return nullptr;
            }
        }
    }

    // Initialize Profiling
    Qnn_ProfileHandle_t profileHandle = nullptr;
    {
        if (ProfilingLevel::OFF != profilingLevel) {
            MLLM_LOG_INFO_LEGACY("Profiling turned on; level = %d", (int)profilingLevel);
            if (ProfilingLevel::BASIC == profilingLevel) {
                MLLM_LOG_INFO_LEGACY("Basic profiling requested. Creating Qnn Profile object.");
                if (QNN_PROFILE_NO_ERROR != qnnInterface.profileCreate(backendHandle, QNN_PROFILE_LEVEL_BASIC, &profileHandle)) {
                    MLLM_LOG_WARN_LEGACY("Unable to create profile handle in the backend.");
                    return nullptr;
                }
            } else if (ProfilingLevel::DETAILED == profilingLevel) {
                MLLM_LOG_INFO_LEGACY("Detailed profiling requested. Creating Qnn Profile object.");
                if (QNN_PROFILE_NO_ERROR != qnnInterface.profileCreate(backendHandle, QNN_PROFILE_LEVEL_DETAILED, &profileHandle)) {
                    MLLM_LOG_ERROR_LEGACY("Unable to create profile handle in the backend.");
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
            {"libQnnLLaMAPackage_CPU.so", "LLaMAPackageInterfaceProvider", "CPU"},
            {"libQnnLLaMAPackage_HTP.so", "LLaMAPackageInterfaceProvider", "HTP"}};

        for (const auto &pkg : opPackages) {
            if (!qnnInterface.backendRegisterOpPackage) {
                MLLM_LOG_ERROR_LEGACY("backendRegisterOpPackageFnHandle is nullptr.");
                return nullptr;
            }
            if (QNN_BACKEND_NO_ERROR != qnnInterface.backendRegisterOpPackage(backendHandle, pkg.path.c_str(), pkg.interfaceProvider.c_str(), pkg.target.c_str())) {
                MLLM_LOG_ERROR_LEGACY("Could not register Op Package: %s and interface provider: %s",
                                      pkg.path.c_str(), pkg.interfaceProvider.c_str());
                return nullptr;
            }
            MLLM_LOG_INFO_LEGACY("Registered Op Package: %s and interface provider: %s",
                                 pkg.path.c_str(), pkg.interfaceProvider.c_str());
        }
    }

    // Create QNN System Interface
    QNN_SYSTEM_INTERFACE_VER_TYPE qnnSystemInterface;
    {
        QnnSystemInterface_t **systemInterfaceProviders{nullptr};
        uint32_t numProviders{0};
        if (QNN_SUCCESS != QnnSystemInterface_getProviders((const QnnSystemInterface_t ***)&systemInterfaceProviders, &numProviders)) {
            MLLM_LOG_ERROR_LEGACY("Failed to get system interface providers.");
            return nullptr;
        }
        if (0 == numProviders) {
            MLLM_LOG_ERROR_LEGACY("Failed to get interface providers: 0 interface providers.");
            return nullptr;
        }
        bool foundValidSystemInterface = false;
        for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
            foundValidSystemInterface = true;
            if (QNN_SYSTEM_API_VERSION_MAJOR == systemInterfaceProviders[pIdx]->systemApiVersion.major && QNN_SYSTEM_API_VERSION_MINOR <= systemInterfaceProviders[pIdx]->systemApiVersion.minor) {
                qnnSystemInterface = systemInterfaceProviders[pIdx]->QNN_SYSTEM_INTERFACE_VER_NAME;
                break;
            }
        }
        if (!foundValidSystemInterface) {
            MLLM_LOG_ERROR_LEGACY("Unable to find a valid system interface.");
            return nullptr;
        }
    }

    return new QNNRuntime(qnnInterface, qnnSystemInterface, logHandle, backendHandle, deviceHandle, profileHandle);
}

bool QNNRuntime::createContext(Qnn_ContextHandle_t &context, QnnContext_Config_t **contextConfig) {
    if (QNN_CONTEXT_NO_ERROR != qnnInterface.contextCreate(backendHandle, deviceHandle, (const QnnContext_Config_t **)&contextConfig, &context)) {
        MLLM_LOG_ERROR("Could not create context");
        return false;
    }
    return true;
}
bool QNNRuntime::retrieveContext(Qnn_ContextHandle_t &context,
                                 std::vector<GraphInfo_t *> &graphsInfo,
                                 QnnContext_Config_t **contextConfig) {
    // Read the binary from qnn_context.bin and get the size in byte
    std::ifstream file("qnn_context.bin", std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    shared_ptr<uint8_t> binaryBuffer(new uint8_t[size], std::default_delete<uint8_t[]>());

    file.read(reinterpret_cast<char *>(binaryBuffer.get()), size);
    file.close();

    // inspect binary info
    QnnSystemContext_Handle_t sysCtxHandle{nullptr};
    if (QNN_SUCCESS != qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
        MLLM_LOG_ERROR("Could not create system handle.");
        return false;
    }
    const QnnSystemContext_BinaryInfo_t *binaryInfo{nullptr};
    Qnn_ContextBinarySize_t binaryInfoSize{0};
    if (QNN_SUCCESS != qnnSystemInterface.systemContextGetBinaryInfo(sysCtxHandle, static_cast<void *>(binaryBuffer.get()), size, &binaryInfo, &binaryInfoSize)) {
        MLLM_LOG_ERROR("Failed to get context binary info");
        return false;
    }

    GraphInfo_t **tmpGraphsInfo = nullptr;
    uint32_t graphNum;
    // fill GraphInfo_t based on binary info
    if (!copyMetadataToGraphsInfo(binaryInfo, tmpGraphsInfo, graphNum)) {
        MLLM_LOG_ERROR("Failed to copy metadata.");
        return false;
    }
    qnnSystemInterface.systemContextFree(sysCtxHandle);
    sysCtxHandle = nullptr;

    graphsInfo.assign(tmpGraphsInfo, tmpGraphsInfo + graphNum);

    Qnn_ContextBinarySize_t writtenSize = 0;
    qnnInterface.contextCreateFromBinary(backendHandle, deviceHandle, (const QnnContext_Config_t **)contextConfig, binaryBuffer.get(), size, &context, profileHandle);

    for (auto &g : graphsInfo) {
        if (QNN_SUCCESS != qnnInterface.graphRetrieve(context, g->graphName, &g->graph)) {
            MLLM_LOG_ERROR("Unable to retrieve graph handle");
            return false;
        }
    }

    MLLM_LOG_INFO_STREAM << "QNN context retrieved from qnn_context.bin";
    return true;
}

} // namespace mllm