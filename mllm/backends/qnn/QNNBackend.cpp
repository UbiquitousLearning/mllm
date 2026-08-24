#include <array>
#include <cerrno>
#include <cstdint>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

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

namespace {

int configuredQnnSequenceTile() {
    const char *value = std::getenv("MLLM_QNN_SEQUENCE_TILE");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
        return 0;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed <= 0
        || parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            std::string("invalid MLLM_QNN_SEQUENCE_TILE: ") + value);
    }
    return static_cast<int>(parsed);
}

size_t qnnBoundaryElementBytes(const Tensor &tensor) {
    switch (tensor.dtype()) {
    case MLLM_TYPE_F32:
        return sizeof(float);
    case MLLM_TYPE_F16:
        return sizeof(mllm_fp16_t);
    case MLLM_TYPE_I32:
        return sizeof(int32_t);
    case MLLM_TYPE_I16:
        return sizeof(int16_t);
    case MLLM_TYPE_I8:
    case MLLM_TYPE_Q8_PER_TENSOR:
        return sizeof(int8_t);
    default:
        throw std::runtime_error(
            "QNN sequence tiling only supports scalar graph-boundary dtypes");
    }
}

void copyQnnSequenceSlice(Tensor &destination, int destination_sequence,
                          Tensor &source, int source_sequence,
                          int sequence_count) {
    if (destination.dtype() != source.dtype()
        || destination.ctype() != source.ctype()
        || destination.batch() != source.batch()
        || destination.head() != source.head()
        || destination.dimension() != source.dimension()
        || destination_sequence < 0 || source_sequence < 0
        || sequence_count <= 0
        || destination_sequence + sequence_count > destination.sequence()
        || source_sequence + sequence_count > source.sequence()) {
        throw std::runtime_error("incompatible QNN sequence-tile copy");
    }
    if (destination.rawHostPtr() == nullptr || source.rawHostPtr() == nullptr) {
        throw std::runtime_error("QNN sequence-tile buffer is not allocated");
    }

    const size_t element_bytes = qnnBoundaryElementBytes(source);
    auto *destination_bytes =
        static_cast<uint8_t *>(destination.rawHostPtr());
    const auto *source_bytes =
        static_cast<const uint8_t *>(source.rawHostPtr());
    const int batches = source.batch();
    const int heads = source.head();
    const int dimensions = source.dimension();

    auto copy_elements = [&](uint64_t destination_offset,
                             uint64_t source_offset,
                             size_t elements) {
        std::memcpy(destination_bytes + destination_offset * element_bytes,
                    source_bytes + source_offset * element_bytes,
                    elements * element_bytes);
    };

    switch (source.ctype()) {
    case BSHD:
        for (int batch = 0; batch < batches; ++batch) {
            copy_elements(
                destination.offset(batch, 0, destination_sequence, 0),
                source.offset(batch, 0, source_sequence, 0),
                static_cast<size_t>(sequence_count) * heads * dimensions);
        }
        break;
    case BHSD:
        for (int batch = 0; batch < batches; ++batch) {
            for (int head = 0; head < heads; ++head) {
                copy_elements(
                    destination.offset(batch, head, destination_sequence, 0),
                    source.offset(batch, head, source_sequence, 0),
                    static_cast<size_t>(sequence_count) * dimensions);
            }
        }
        break;
    case SBHD:
        copy_elements(
            destination.offset(0, 0, destination_sequence, 0),
            source.offset(0, 0, source_sequence, 0),
            static_cast<size_t>(sequence_count) * batches * heads
                * dimensions);
        break;
    case BHDS:
        for (int batch = 0; batch < batches; ++batch) {
            for (int head = 0; head < heads; ++head) {
                for (int dimension = 0; dimension < dimensions;
                     ++dimension) {
                    copy_elements(
                        destination.offset(batch, head,
                                           destination_sequence, dimension),
                        source.offset(batch, head, source_sequence,
                                      dimension),
                        sequence_count);
                }
            }
        }
        break;
    case BDHS:
    case DBHS:
        for (int batch = 0; batch < batches; ++batch) {
            for (int head = 0; head < heads; ++head) {
                for (int dimension = 0; dimension < dimensions;
                     ++dimension) {
                    for (int sequence = 0; sequence < sequence_count;
                         ++sequence) {
                        copy_elements(
                            destination.offset(
                                batch, head,
                                destination_sequence + sequence, dimension),
                            source.offset(batch, head,
                                          source_sequence + sequence,
                                          dimension),
                            1);
                    }
                }
            }
        }
        break;
    default:
        throw std::runtime_error(
            "unsupported graph-boundary layout for QNN sequence tiling");
    }
}

// PhoneLM factorizes the token dimension before its wide linear operators.
// For example, a logical M512 tensor is represented as [1,32,16,D] in QNN's
// physical BSHD order while the cached M256 graph uses [1,16,16,D]. Both are
// contiguous arrays of token rows, so a tile is a byte-contiguous range even
// though neither individual logical axis is named SEQUENCE=256.
void copyQnnFlatTokenSlice(Tensor &destination, int destination_token,
                           Tensor &source, int source_token,
                           int token_count) {
    const int destination_tokens =
        destination.head() * destination.sequence();
    const int source_tokens = source.head() * source.sequence();
    if (destination.dtype() != source.dtype()
        || destination.ctype() != BSHD || source.ctype() != BSHD
        || destination.batch() != source.batch()
        || destination.dimension() != source.dimension()
        || destination_token < 0 || source_token < 0 || token_count <= 0
        || destination_token + token_count > destination_tokens
        || source_token + token_count > source_tokens) {
        throw std::runtime_error(
            "incompatible flat-token QNN sequence-tile copy");
    }
    if (destination.rawHostPtr() == nullptr || source.rawHostPtr() == nullptr) {
        throw std::runtime_error("QNN sequence-tile buffer is not allocated");
    }

    const size_t element_bytes = qnnBoundaryElementBytes(source);
    const size_t token_bytes =
        static_cast<size_t>(source.dimension()) * element_bytes;
    const size_t destination_batch_bytes =
        static_cast<size_t>(destination_tokens) * token_bytes;
    const size_t source_batch_bytes =
        static_cast<size_t>(source_tokens) * token_bytes;
    auto *destination_bytes =
        static_cast<uint8_t *>(destination.rawHostPtr());
    const auto *source_bytes =
        static_cast<const uint8_t *>(source.rawHostPtr());
    for (int batch = 0; batch < source.batch(); ++batch) {
        std::memcpy(
            destination_bytes
                + static_cast<size_t>(batch) * destination_batch_bytes
                + static_cast<size_t>(destination_token) * token_bytes,
            source_bytes + static_cast<size_t>(batch) * source_batch_bytes
                + static_cast<size_t>(source_token) * token_bytes,
            static_cast<size_t>(token_count) * token_bytes);
    }
}

} // namespace

namespace {
ProfilingLevel default_profiling_level = ProfilingLevel::DETAILED;
} // namespace

void QNNBackend::setDefaultProfilingLevel(ProfilingLevel level) {
    default_profiling_level = level;
}

ProfilingLevel QNNBackend::defaultProfilingLevel() {
    return default_profiling_level;
}

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
    m_profilingLevel = defaultProfilingLevel();
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
        const char *prepare_only_env =
            std::getenv("MLLM_QNN_CONTEXT_PREPARE_ONLY");
        const bool prepare_only = prepare_only_env != nullptr
            && std::strcmp(prepare_only_env, "0") != 0;
        if (prepare_only) {
            QnnHtpContext_CustomConfig_t htp_config =
                QNN_HTP_CONTEXT_CUSTOM_CONFIG_INIT;
            htp_config.option =
                QNN_HTP_CONTEXT_CONFIG_OPTION_PREPARE_ONLY;
            htp_config.isPrepareOnly = true;
            QnnContext_Config_t context_config = QNN_CONTEXT_CONFIG_INIT;
            context_config.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
            context_config.customConfig = &htp_config;
            const QnnContext_Config_t *context_configs[] = {
                &context_config, nullptr};
            contextStatus = mRuntime->createContext(m_context,
                                                    context_configs);
            if (contextStatus) {
                MLLM_LOG_INFO("QNN context prepare-only mode enabled");
            }
        } else {
            contextStatus = mRuntime->createContext(m_context, nullptr);
        }
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

    // Sequence-tile scratch tensors allocate through this backend. Release
    // them while the QNN memory manager and its rpcmem allocator are still
    // alive; otherwise member destruction after mem_manager_.reset() would
    // dereference a dead allocator.
    sequenceTileGraphs_.clear();
    sequenceTilePendingInputs_.clear();
    sequenceTileBufferPool_.clear();

    // QNN resources must be released while the dynamically loaded backend is
    // still alive. Registered memory depends on the context, and the context
    // in turn depends on the runtime's device and backend handles.
    mPerf.reset();
    mem_manager_.reset();
    if (mRuntime != nullptr && m_context != nullptr) {
        CALL_QNN(mRuntime->qnnInterface.contextFree(m_context, mRuntime->profileHandle));
        m_context = nullptr;
    }
    mRuntime.reset();
}

void QNNBackend::onSetUpStart(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName) {
    if (configuredQnnSequenceTile() != 0) {
        sequenceTilePendingInputs_[graphName] = inputs;
    }
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

    QnnHtpGraph_CustomConfig_t shareIoConfigInfo;
    shareIoConfigInfo.option =
        QNN_HTP_GRAPH_CONFIG_OPTION_SHARE_IO_BUFFER;
    shareIoConfigInfo.shareIOBuffer = true;
    QnnGraph_Config_t shareIoConfig;
    shareIoConfig.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    shareIoConfig.customConfig = &shareIoConfigInfo;

    const char *share_io_env =
        std::getenv("MLLM_QNN_SHARE_IO_BUFFER");
    const bool share_io = share_io_env != nullptr
        && share_io_env[0] != '\0'
        && std::strcmp(share_io_env, "0") != 0;
    const QnnGraph_Config_t *graphConfigList[] = {
        &vtcmConfig, &slcConfig, share_io ? &shareIoConfig : nullptr,
        nullptr};

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

    // Context generation only needs finalized graph metadata. Registering all
    // traced graph I/O buffers keeps large M=512 activation mappings alive
    // until serialization and can exhaust the DSP address space before the
    // final layers are prepared.
    if (!isFromCache
        && std::getenv("MLLM_QNN_CONTEXT_GENERATE_ONLY") != nullptr) {
        return;
    }

    auto qnnMM = std::static_pointer_cast<QNNMemoryManager>(mem_manager_);

    const int configured_tile_sequence = configuredQnnSequenceTile();
    if (configured_tile_sequence != 0) {
        if (!isFromCache) {
            throw std::runtime_error(
                "MLLM_QNN_SEQUENCE_TILE requires a cached QNN context");
        }
        const std::string effective_graph_name =
            graphInfo->graphName == nullptr ? std::string() : graphInfo->graphName;
        const auto pending_it =
            sequenceTilePendingInputs_.find(effective_graph_name);
        if (pending_it == sequenceTilePendingInputs_.end()) {
            throw std::runtime_error(
                "missing logical inputs for QNN sequence-tiled graph "
                + effective_graph_name);
        }
        if (pending_it->second.size() != graphInfo->numInputTensors
            || inputs.size() != graphInfo->numOutputTensors) {
            throw std::runtime_error(
                "QNN sequence-tile boundary count mismatch for graph "
                + effective_graph_name);
        }

        SequenceTileGraphState tile_state;
        tile_state.tile_sequence = configured_tile_sequence;
        auto order_boundaries = [&effective_graph_name](
                                    const std::vector<std::shared_ptr<Tensor>>
                                        &candidates,
                                    Qnn_Tensor_t *qnn_tensors,
                                    size_t tensor_count,
                                    const std::vector<uint8_t *> &buffers,
                                    const char *direction) {
            std::vector<std::shared_ptr<Tensor>> ordered;
            std::vector<bool> used(candidates.size(), false);
            ordered.reserve(tensor_count);
            for (size_t index = 0; index < tensor_count; ++index) {
                size_t match = candidates.size();
                const char *qnn_name = qnn_tensors[index].v1.name;
                if (qnn_name != nullptr) {
                    for (size_t candidate = 0; candidate < candidates.size();
                         ++candidate) {
                        if (!used[candidate]
                            && candidates[candidate]->name() == qnn_name) {
                            match = candidate;
                            break;
                        }
                    }
                }
                if (match == candidates.size() && index < buffers.size()) {
                    for (size_t candidate = 0; candidate < candidates.size();
                         ++candidate) {
                        if (!used[candidate]
                            && candidates[candidate]->rawHostPtr()
                                == buffers[index]) {
                            match = candidate;
                            break;
                        }
                    }
                }
                if (match == candidates.size()) {
                    throw std::runtime_error(
                        "cannot map QNN sequence-tile "
                        + std::string(direction) + std::to_string(index)
                        + " for graph " + effective_graph_name);
                }
                used[match] = true;
                ordered.push_back(candidates[match]);
            }
            return ordered;
        };
        tile_state.logical_inputs = order_boundaries(
            pending_it->second, qnnInputs, graphInfo->numInputTensors,
            *currentInputBuffers, "input");
        tile_state.logical_outputs = order_boundaries(
            inputs, qnnOutputs, graphInfo->numOutputTensors,
            *currentOutputBuffers, "output");

        auto prepare_boundary = [&](const std::shared_ptr<Tensor> &logical,
                                    Qnn_Tensor_t &qnn_tensor,
                                    const char *direction, size_t slot,
                                    bool &flat_token_boundary) {
            if (qnn_tensor.v1.rank != 4
                || qnn_tensor.v1.dimensions == nullptr) {
                throw std::runtime_error(
                    "QNN sequence tiling requires rank-4 graph boundaries");
            }
            const int logical_batch = logical->batch();
            const int logical_head = logical->head();
            const int logical_sequence = logical->sequence();
            const int logical_dimension = logical->dimension();
            std::array<int, 4> expected_dimensions{};
            size_t sequence_axis = 0;
            switch (logical->ctype()) {
            case BSHD:
                sequence_axis = 1;
                expected_dimensions = {logical_batch,
                                       configured_tile_sequence,
                                       logical_head, logical_dimension};
                break;
            case BHSD:
                sequence_axis = 2;
                expected_dimensions = {logical_batch, logical_head,
                                       configured_tile_sequence,
                                       logical_dimension};
                break;
            case BHDS:
                sequence_axis = 3;
                expected_dimensions = {logical_batch, logical_head,
                                       logical_dimension,
                                       configured_tile_sequence};
                break;
            case BDHS:
                sequence_axis = 3;
                expected_dimensions = {logical_batch, logical_dimension,
                                       logical_head,
                                       configured_tile_sequence};
                break;
            case SBHD:
                sequence_axis = 0;
                expected_dimensions = {configured_tile_sequence,
                                       logical_batch, logical_head,
                                       logical_dimension};
                break;
            case DBHS:
                sequence_axis = 3;
                expected_dimensions = {logical_dimension, logical_batch,
                                       logical_head,
                                       configured_tile_sequence};
                break;
            default:
                throw std::runtime_error(
                    "unsupported QNN sequence-tile boundary layout in graph "
                    + effective_graph_name);
            }
            uint64_t metadata_elements = 1;
            for (size_t dimension = 0; dimension < 4; ++dimension) {
                metadata_elements *= qnn_tensor.v1.dimensions[dimension];
            }
            const uint64_t expected_elements =
                static_cast<uint64_t>(logical_batch) * logical_head
                * configured_tile_sequence * logical_dimension;
            // Cached QNN graphs may flatten H*D into a single physical
            // dimension (for example [B,S,1,H*D]).  That is byte-compatible
            // with BSHD and must not be rejected merely because the logical
            // view restores H and D separately.
            const bool metadata_matches =
                qnn_tensor.v1.dimensions[sequence_axis]
                    == configured_tile_sequence
                && metadata_elements == expected_elements;
            const bool sequence_matches = metadata_matches
                && logical_sequence > configured_tile_sequence
                && logical_sequence % configured_tile_sequence == 0;
            const uint64_t logical_flat_tokens =
                static_cast<uint64_t>(logical_head) * logical_sequence;
            const uint64_t qnn_flat_tokens =
                static_cast<uint64_t>(qnn_tensor.v1.dimensions[1])
                * qnn_tensor.v1.dimensions[2];
            const bool flat_token_matches = logical->ctype() == BSHD
                && qnn_tensor.v1.dimensions[0]
                    == static_cast<uint32_t>(logical_batch)
                && qnn_tensor.v1.dimensions[3]
                    == static_cast<uint32_t>(logical_dimension)
                && qnn_flat_tokens
                    == static_cast<uint64_t>(configured_tile_sequence)
                && metadata_elements
                    == static_cast<uint64_t>(logical_batch)
                        * configured_tile_sequence * logical_dimension
                && logical_flat_tokens
                    > static_cast<uint64_t>(configured_tile_sequence)
                && logical_flat_tokens % configured_tile_sequence == 0;
            flat_token_boundary = !sequence_matches && flat_token_matches;
            if (!sequence_matches && !flat_token_matches) {
                std::string detail;
                for (size_t dimension = 0; dimension < 4; ++dimension) {
                    detail += (dimension == 0 ? " qnn=[" : ",")
                        + std::to_string(
                            qnn_tensor.v1.dimensions[dimension]);
                }
                detail += "] expected=[";
                for (size_t dimension = 0; dimension < 4; ++dimension) {
                    detail += (dimension == 0 ? "" : ",")
                        + std::to_string(expected_dimensions[dimension]);
                }
                detail += "] logical_bhsd=["
                    + std::to_string(logical_batch) + ","
                    + std::to_string(logical_head) + ","
                    + std::to_string(logical_sequence) + ","
                    + std::to_string(logical_dimension) + "] ctype="
                    + std::to_string(static_cast<int>(logical->ctype()));
                throw std::runtime_error(
                    "QNN sequence-tile shape mismatch for graph "
                    + effective_graph_name + " boundary " + direction
                    + std::to_string(slot) + detail);
            }
            const int boundary_tile_count = flat_token_boundary
                ? static_cast<int>(logical_flat_tokens
                                   / configured_tile_sequence)
                : logical_sequence / configured_tile_sequence;
            if (tile_state.tile_count == 0) {
                tile_state.tile_count = boundary_tile_count;
            } else if (tile_state.tile_count != boundary_tile_count) {
                throw std::runtime_error(
                    "inconsistent QNN sequence-tile count in graph "
                    + effective_graph_name);
            }

            const int tile_head = flat_token_boundary
                ? static_cast<int>(qnn_tensor.v1.dimensions[2])
                : logical_head;
            const int tile_sequence = flat_token_boundary
                ? static_cast<int>(qnn_tensor.v1.dimensions[1])
                : configured_tile_sequence;
            const std::string pool_key =
                std::string(direction) + std::to_string(slot) + ":"
                + std::to_string(static_cast<int>(logical->dtype())) + ":"
                + std::to_string(static_cast<int>(logical->ctype())) + ":"
                + std::to_string(logical_batch) + ":"
                + std::to_string(tile_head) + ":"
                + std::to_string(tile_sequence) + ":"
                + std::to_string(logical_dimension);
            auto pool_it = sequenceTileBufferPool_.find(pool_key);
            if (pool_it == sequenceTileBufferPool_.end()) {
                auto tile = std::make_shared<Tensor>(this);
                tile->setName("qnn_sequence_tile_" + pool_key);
                tile->setDtype(logical->dtype());
                tile->setCtype(logical->ctype());
                tile->reshape(logical_batch, tile_head, tile_sequence,
                              logical_dimension);
                tile->alloc();
                sequenceTileScratchBytes_ += tile->cntSize();
                pool_it = sequenceTileBufferPool_
                              .emplace(pool_key, std::move(tile))
                              .first;
            }
            qnn_tensor.v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            qnnMM->registerQnnTensor(pool_it->second->rawHostPtr(),
                                     qnn_tensor);
            return pool_it->second;
        };

        for (size_t index = 0; index < graphInfo->numInputTensors; ++index) {
            bool flat_token_boundary = false;
            tile_state.tile_inputs.push_back(prepare_boundary(
                tile_state.logical_inputs[index], qnnInputs[index], "input",
                index, flat_token_boundary));
            tile_state.flat_token_inputs.push_back(flat_token_boundary);
        }
        for (size_t index = 0; index < graphInfo->numOutputTensors; ++index) {
            bool flat_token_boundary = false;
            tile_state.tile_outputs.push_back(prepare_boundary(
                tile_state.logical_outputs[index], qnnOutputs[index],
                "output", index, flat_token_boundary));
            tile_state.flat_token_outputs.push_back(flat_token_boundary);
        }
        sequenceTileGraphs_[effective_graph_name] = std::move(tile_state);
        std::cout << "QNN_SEQUENCE_TILE_SETUP graph="
                  << effective_graph_name
                  << " logical_sequence=" << inputs.front()->sequence()
                  << " tile_sequence=" << configured_tile_sequence
                  << " tiles="
                  << sequenceTileGraphs_[effective_graph_name].tile_count
                  << " scratch_pool_bytes=" << sequenceTileScratchBytes_
                  << std::endl;
        return;
    }

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
    auto execute_graph = [&]() {
        if (mRuntime->qnnInterface.graphExecute(
                graphInfo->graph, graphInfo->inputTensors,
                graphInfo->numInputTensors, graphInfo->outputTensors,
                graphInfo->numOutputTensors, mRuntime->profileHandle, nullptr)
            != QNN_GRAPH_NO_ERROR) {
            MLLM_LOG_ERROR_STREAM
                << "Error in executing graph: " << graphName << std::endl;
        }
    };

    const auto tile_it = sequenceTileGraphs_.find(graphName);
    if (tile_it == sequenceTileGraphs_.end()) {
        execute_graph();
    } else {
        auto &tile_state = tile_it->second;
        ++sequenceTileProfile_.graph_calls;
        for (int tile_index = 0; tile_index < tile_state.tile_count;
             ++tile_index) {
            uint64_t stage_begin = mllm_time_us();
            for (size_t input_index = 0;
                 input_index < tile_state.logical_inputs.size();
                 ++input_index) {
                if (tile_state.flat_token_inputs[input_index]) {
                    copyQnnFlatTokenSlice(
                        *tile_state.tile_inputs[input_index], 0,
                        *tile_state.logical_inputs[input_index],
                        tile_index * tile_state.tile_sequence,
                        tile_state.tile_sequence);
                } else {
                    copyQnnSequenceSlice(
                        *tile_state.tile_inputs[input_index], 0,
                        *tile_state.logical_inputs[input_index],
                        tile_index * tile_state.tile_sequence,
                        tile_state.tile_sequence);
                }
            }
            sequenceTileProfile_.input_copy_us +=
                mllm_time_us() - stage_begin;

            stage_begin = mllm_time_us();
            execute_graph();
            sequenceTileProfile_.execute_us +=
                mllm_time_us() - stage_begin;

            stage_begin = mllm_time_us();
            for (size_t output_index = 0;
                 output_index < tile_state.logical_outputs.size();
                 ++output_index) {
                if (tile_state.flat_token_outputs[output_index]) {
                    copyQnnFlatTokenSlice(
                        *tile_state.logical_outputs[output_index],
                        tile_index * tile_state.tile_sequence,
                        *tile_state.tile_outputs[output_index], 0,
                        tile_state.tile_sequence);
                } else {
                    copyQnnSequenceSlice(
                        *tile_state.logical_outputs[output_index],
                        tile_index * tile_state.tile_sequence,
                        *tile_state.tile_outputs[output_index], 0,
                        tile_state.tile_sequence);
                }
            }
            sequenceTileProfile_.output_copy_us +=
                mllm_time_us() - stage_begin;
            ++sequenceTileProfile_.tile_calls;
        }
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

bool QNNBackend::saveQNNContext() {
    uint64_t binarySize, writtenSize;

    const auto size_status =
        mRuntime->qnnInterface.contextGetBinarySize(m_context, &binarySize);
    if (size_status != QNN_CONTEXT_NO_ERROR || binarySize == 0) {
        MLLM_LOG_ERROR_STREAM << "Failed to query QNN context binary size: "
                              << size_status << std::endl;
        return false;
    }

    std::unique_ptr<uint8_t[]> binaryBuffer(new uint8_t[binarySize]);

    const auto binary_status = mRuntime->qnnInterface.contextGetBinary(
        m_context, reinterpret_cast<void *>(binaryBuffer.get()), binarySize,
        &writtenSize);
    if (binary_status != QNN_CONTEXT_NO_ERROR || writtenSize == 0
        || writtenSize > binarySize) {
        MLLM_LOG_ERROR_STREAM << "Failed to serialize QNN context: status "
                              << binary_status << ", capacity " << binarySize
                              << ", written " << writtenSize << std::endl;
        return false;
    }

    std::ofstream file("qnn_context.bin", std::ios::binary);
    file.write(reinterpret_cast<char *>(binaryBuffer.get()), writtenSize);
    if (!file.good()) {
        MLLM_LOG_ERROR_STREAM << "Failed to write qnn_context.bin" << std::endl;
        return false;
    }
    file.close();

    std::cout << "QNN context saved to qnn_context.bin written " << writtenSize << std::endl;
    return true;
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
                    layer->owns_op_ = false;
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

    // A null device is valid for context creation and means that QNN manages
    // the default device internally, so there is no explicit handle to free.
    if (deviceHandle != nullptr) {
        CALL_QNN(qnnInterface.deviceFree(deviceHandle));
    }

    // Free Backend
    if (backendHandle != nullptr) {
        CALL_QNN(qnnInterface.backendFree(backendHandle));
    }

    // Free Log
    if (logHandle != nullptr) {
        CALL_QNN(qnnInterface.logFree(logHandle));
    }
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

        const char *htpOpPackageEnv =
            std::getenv("MLLM_QNN_HTP_OP_PACKAGE");
        const std::string htpOpPackage =
            htpOpPackageEnv != nullptr && htpOpPackageEnv[0] != '\0'
                ? htpOpPackageEnv
                : "libQnnLLaMAPackage_HTP.so";

        std::vector<OpPackageInfo> opPackages = {
            {"libQnnLLaMAPackage_CPU.so", "LLaMAPackageInterfaceProvider", "CPU"},
            {htpOpPackage, "LLaMAPackageInterfaceProvider", "HTP"}};

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

bool QNNRuntime::createContext(
    Qnn_ContextHandle_t &context,
    const QnnContext_Config_t **contextConfig) {
    if (QNN_CONTEXT_NO_ERROR
        != qnnInterface.contextCreate(backendHandle, deviceHandle,
                                      contextConfig, &context)) {
        MLLM_LOG_ERROR("Could not create context");
        return false;
    }
    return true;
}
bool QNNRuntime::retrieveContext(Qnn_ContextHandle_t &context,
                                 std::vector<GraphInfo_t *> &graphsInfo,
                                 const QnnContext_Config_t **contextConfig) {
    size_t size = 0;
#if defined(__linux__)
    const int context_fd = open("qnn_context.bin", O_RDONLY | O_CLOEXEC);
    if (context_fd < 0) {
        MLLM_LOG_ERROR_STREAM << "Failed to open qnn_context.bin: "
                              << std::strerror(errno) << std::endl;
        return false;
    }
    struct stat context_stat {};
    if (fstat(context_fd, &context_stat) != 0 || context_stat.st_size <= 0
        || static_cast<uint64_t>(context_stat.st_size)
            > std::numeric_limits<size_t>::max()) {
        MLLM_LOG_ERROR("QNN context binary has an invalid size");
        close(context_fd);
        return false;
    }
    size = static_cast<size_t>(context_stat.st_size);
    int mmap_flags = MAP_PRIVATE;
#if defined(MAP_POPULATE)
    // Match the old read() path's warm-cache behavior without creating an
    // anonymous 1+ GiB duplicate. Prefaulting happens before TTFT timing.
    mmap_flags |= MAP_POPULATE;
#endif
    void *mapped_context = mmap(nullptr, size, PROT_READ, mmap_flags,
                                context_fd, 0);
    const int mmap_error = errno;
    close(context_fd);
    if (mapped_context == MAP_FAILED) {
        MLLM_LOG_ERROR_STREAM << "Failed to mmap qnn_context.bin: "
                              << std::strerror(mmap_error) << std::endl;
        return false;
    }
    contextBinaryBuffer = shared_ptr<uint8_t>(
        static_cast<uint8_t *>(mapped_context),
        [size](uint8_t *mapping) {
            if (mapping != nullptr) {
                munmap(mapping, size);
            }
        });
    MLLM_LOG_INFO_STREAM << "Memory-mapped QNN context binary: " << size
                         << " bytes" << std::endl;
#else
    // Non-Linux fallback for host-side tooling.
    std::ifstream file("qnn_context.bin", std::ios::binary | std::ios::ate);
    const std::streamsize stream_size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (stream_size <= 0) {
        MLLM_LOG_ERROR("QNN context binary is empty");
        return false;
    }
    size = static_cast<size_t>(stream_size);
    contextBinaryBuffer = shared_ptr<uint8_t>(
        new uint8_t[size], std::default_delete<uint8_t[]>());

    file.read(reinterpret_cast<char *>(contextBinaryBuffer.get()), size);
    if (!file.good()) {
        MLLM_LOG_ERROR("Failed to read qnn_context.bin");
        return false;
    }
    file.close();
#endif

    // inspect binary info
    QnnSystemContext_Handle_t sysCtxHandle{nullptr};
    if (QNN_SUCCESS != qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
        MLLM_LOG_ERROR("Could not create system handle.");
        return false;
    }
    const QnnSystemContext_BinaryInfo_t *binaryInfo{nullptr};
    Qnn_ContextBinarySize_t binaryInfoSize{0};
    if (QNN_SUCCESS != qnnSystemInterface.systemContextGetBinaryInfo(
                           sysCtxHandle,
                           static_cast<void *>(contextBinaryBuffer.get()),
                           size, &binaryInfo, &binaryInfoSize)) {
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

    QnnContext_Config_t memory_limit_config = QNN_CONTEXT_CONFIG_INIT;
    QnnContext_Config_t persistent_binary_config = QNN_CONTEXT_CONFIG_INIT;
    std::vector<const QnnContext_Config_t *> effective_configs;
    if (contextConfig != nullptr) {
        for (size_t index = 0; contextConfig[index] != nullptr; ++index) {
            effective_configs.push_back(contextConfig[index]);
        }
    }
    const char *memory_limit_env =
        std::getenv("MLLM_QNN_CONTEXT_MEMORY_LIMIT_MB");
    bool memory_limit_enabled = false;
    if (memory_limit_env != nullptr
        && std::strcmp(memory_limit_env, "0") != 0) {
        char *end = nullptr;
        const unsigned long long memory_limit =
            std::strtoull(memory_limit_env, &end, 10);
        if (end == memory_limit_env || *end != '\0' || memory_limit == 0) {
            MLLM_LOG_ERROR_STREAM
                << "Invalid MLLM_QNN_CONTEXT_MEMORY_LIMIT_MB: "
                << memory_limit_env << std::endl;
            return false;
        }
        memory_limit_config.option = QNN_CONTEXT_CONFIG_MEMORY_LIMIT_HINT;
        memory_limit_config.memoryLimitHint = memory_limit;
        memory_limit_enabled = true;
        effective_configs.push_back(&memory_limit_config);
        MLLM_LOG_INFO_STREAM << "QNN context memory limit hint enabled: "
                             << memory_limit << " MB" << std::endl;
    }
    bool persistent_binary_enabled = memory_limit_enabled;
    const char *persistent_binary_env =
        std::getenv("MLLM_QNN_CONTEXT_PERSISTENT_BINARY");
    if (persistent_binary_env != nullptr
        && persistent_binary_env[0] != '\0') {
        if (std::strcmp(persistent_binary_env, "1") == 0) {
            persistent_binary_enabled = true;
        } else if (std::strcmp(persistent_binary_env, "0") == 0) {
            persistent_binary_enabled = false;
        } else {
            MLLM_LOG_ERROR_STREAM
                << "Invalid MLLM_QNN_CONTEXT_PERSISTENT_BINARY: "
                << persistent_binary_env << std::endl;
            return false;
        }
    }
    if (memory_limit_enabled && !persistent_binary_enabled) {
        MLLM_LOG_ERROR(
            "QNN context memory limit requires a persistent binary");
        return false;
    }
    if (persistent_binary_enabled) {
        persistent_binary_config.option =
            QNN_CONTEXT_CONFIG_PERSISTENT_BINARY;
        persistent_binary_config.isPersistentBinary = 1;
        effective_configs.push_back(&persistent_binary_config);
        if (!memory_limit_enabled) {
            MLLM_LOG_INFO(
                "QNN persistent context binary enabled without a memory "
                "limit hint");
        }
    }
    effective_configs.push_back(nullptr);
    const auto create_status = qnnInterface.contextCreateFromBinary(
        backendHandle, deviceHandle, effective_configs.data(),
        contextBinaryBuffer.get(), size, &context, profileHandle);
    if (create_status != QNN_CONTEXT_NO_ERROR) {
        MLLM_LOG_ERROR_STREAM << "QNN contextCreateFromBinary failed: "
                              << create_status << std::endl;
        return false;
    }

    for (auto &g : graphsInfo) {
        if (QNN_SUCCESS != qnnInterface.graphRetrieve(context, g->graphName, &g->graph)) {
            MLLM_LOG_ERROR("Unable to retrieve graph handle");
            return false;
        }
    }

    if (!persistent_binary_enabled) {
        // QNN owns the deserialized graphs after contextCreateFromBinary.
        // Keeping the anonymous 1+ GiB input cache alive only increases the
        // process RSS unless PERSISTENT_BINARY explicitly promises that the
        // pointer remains readable for the lifetime of the context.
        contextBinaryBuffer.reset();
        MLLM_LOG_INFO("Released non-persistent QNN context binary buffer");
    } else {
#if defined(__linux__)
        const char *madvise_env =
            std::getenv("MLLM_QNN_CONTEXT_MADVISE_DONTNEED");
        const bool release_pages = madvise_env != nullptr
            && std::strcmp(madvise_env, "0") != 0;
        if (release_pages) {
            if (madvise(contextBinaryBuffer.get(), size, MADV_DONTNEED) == 0) {
                // The mapping address remains valid and graph switching faults
                // file pages back in on demand. This minimizes RSS at the cost
                // of additional page faults.
                MLLM_LOG_INFO(
                    "Released resident QNN context pages; persistent mmap "
                    "retained");
            } else {
                MLLM_LOG_ERROR_STREAM
                    << "madvise(MADV_DONTNEED) failed for QNN context: "
                    << std::strerror(errno) << std::endl;
            }
        } else {
            MLLM_LOG_INFO(
                "Persistent QNN context uses reclaimable file-backed mmap");
        }
#endif
    }

    MLLM_LOG_INFO_STREAM << "QNN context retrieved from qnn_context.bin";
    return true;
}

} // namespace mllm
