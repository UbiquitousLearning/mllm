#include "NNAPIBackend.hpp"
#include "Types.hpp"
#include <cstdint>

// TODO: float <--> half convert for armv82
#define FLOAT_TO_HALF(...)
#define HALF_TO_FLOAT(...)

#define NNAPI_CHECK(func, ...)                                   \
    do {                                                         \
        const auto _status = (func(__VA_ARGS__));                \
        if (_status != ANEURALNETWORKS_NO_ERROR) {               \
            const auto ENUM_TO_STR = NNAPIEnumToString(_status); \
            exit(0);                                             \
        }                                                        \
    } while (0)

namespace mllm {

std::string NNAPIEnumToString(int code) {
    switch (code) {
#define _ENUM_TO_STR(code) \
    case ANEURALNETWORKS_##code: return #code
        // ResultCode begin
        _ENUM_TO_STR(NO_ERROR);
        _ENUM_TO_STR(OUT_OF_MEMORY);
        _ENUM_TO_STR(INCOMPLETE);
        _ENUM_TO_STR(UNEXPECTED_NULL);
        _ENUM_TO_STR(BAD_DATA);
        _ENUM_TO_STR(OP_FAILED);
        _ENUM_TO_STR(BAD_STATE);
        _ENUM_TO_STR(UNMAPPABLE);
        _ENUM_TO_STR(OUTPUT_INSUFFICIENT_SIZE);
        _ENUM_TO_STR(UNAVAILABLE_DEVICE);
    // ResultCode end
    default:
        return "UNKNOWN_ENUM";
#undef ENUM_TO_STR
    }
}

NNAPIBackend::NNAPIBackend(shared_ptr<MemoryManager> mm) :
    Backend(mm) {
    initCreatorMap();
    registerOps();

    if (mNNAPIModel_ == nullptr) {
        NNAPI_CHECK(ANeuralNetworksModel_create_27, &mNNAPIModel_);
    }
    if (mNNAPIDevices_.empty()) {
        uint32_t numDevices = 0;
        NNAPI_CHECK(ANeuralNetworks_getDeviceCount_29, &numDevices);
        mNNAPIDevices_.resize(numDevices);
        // NNAPI_DEVICE_LOG("[NNAPI] numDevices = %d\n", numDevices);
        for (int i = 0; i < numDevices; i++) {
            NNAPI_CHECK(ANeuralNetworks_getDevice_29, i, &mNNAPIDevices_[i].device);
            NNAPI_CHECK(ANeuralNetworksDevice_getName_29, mNNAPIDevices_[i].device, &mNNAPIDevices_[i].name);
            NNAPI_CHECK(ANeuralNetworksDevice_getType_29, mNNAPIDevices_[i].device, &mNNAPIDevices_[i].type);
        }
    }
}

NNAPIBackend::~NNAPIBackend() {
    ANeuralNetworksCompilation_free_27(mNNAPICompilation_);
    ANeuralNetworksModel_free_27(mNNAPIModel_);
}

Op *NNAPIBackend::opCreate(const OpParam &op_param) {
    OpType optype = OpType(op_param.find("type")->second);
    auto *map = map_creator_;
    auto iter = map->find(optype);
    if (iter == map->end()) {
        printf("Don't support type \n");
        return nullptr;
    }
    Op *exe = nullptr;
    exe = iter->second->create(op_param, this);
    return exe;
    // return nullptr;
}

void NNAPIBackend::registerOps() {
    // ADD,
    // CAUSALMASK,
    // MATMUL,
    // RMSNORM,
    // ROPE,
    // SCALE,
    // SILU,
    // SOFTMAX
    // addCreator(ADD, (NNAPIBackend::Creator *)(new NNAPIAddCreator()));
    // addCreator(CAUSALMASK, (NNAPIBackend::Creator *)(new NNAPICausalMaskCreator()));
    // addCreator(MATMUL, (NNAPIBackend::Creator *)(new NNAPIMatmulCreator()));
    // addCreator(RMSNORM, (NNAPIBackend::Creator *)(new NNAPIRMSNormCreator()));
    // addCreator(ROPE, (NNAPIBackend::Creator *)(new NNAPIRoPECreator()));
    // addCreator(SCALE, (NNAPIBackend::Creator *)(new NNAPIScaleCreator()));
    // addCreator(SILU, (NNAPIBackend::Creator *)(new NNAPISiLUCreator()));
    // addCreator(SOFTMAX, (NNAPIBackend::Creator *)(new NNAPISoftMaxCreator()));
    // addCreator(LINEAR, (NNAPIBackend::Creator *)(new NNAPILinearCreator()));
}

uint32_t NNAPIBackend::getTensorIdx(const Tensor *t, bool dequant) {
    if (dequant) {
        const auto &qiter = dequantIdxMap_.find(t);
        if (qiter != dequantIdxMap_.end()) {
            return qiter->second;
        }
    }
    const auto &iter = tensorIdxMap_.find(t);
    if (iter != tensorIdxMap_.end()) {
        return iter->second;
    }
    std::vector<uint32_t> udims;
    for (auto d : t->shape()) {
        udims.push_back(d);
    }
    // scalar shape is {1} in NNAPI
    if (udims.empty()) {
        udims.push_back(1);
    }
    float scale = 0.F;
    int zero = 0;
    auto byteWidth = t->byteWidth();
    // TODO: ANEURALNETWORKS_TENSOR_INT32 and ANEURALNETWORKS_TENSOR_QUANT8_ASYMM
    auto code = ANEURALNETWORKS_TENSOR_FLOAT32;
    uint32_t idx = -1;
    // TODO: CONSTANT describe in tensor
    idx = buildOperand(nullptr, 0, code, udims, &scale, zero);
    tensorIdxMap_.insert(std::make_pair(t, idx));
    return idx;
}

uint32_t NNAPIBackend::buildOperand(const void *data, size_t size, OperandCode code, std::vector<uint32_t> dims, const float *scales, int zero) {
    // TODO: check if determined by byteWidth in tensor
    bool useFP16 = (bytes() == 2 && code == ANEURALNETWORKS_TENSOR_FLOAT32);
    if (useFP16) {
        code = ANEURALNETWORKS_TENSOR_FLOAT16;
        size /= 2;
    }
    float scale = ((scales != nullptr) && code != ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL) ? *scales : 0.F;
    ANeuralNetworksOperandType operandType;
    operandType.type = code;
    operandType.dimensionCount = static_cast<uint32_t>(dims.size());
    operandType.dimensions = dims.empty() ? nullptr : dims.data();
    operandType.scale = scale;
    operandType.zeroPoint = zero;

    uint32_t operandIdx = tensorIdx_++;
#ifdef DEBUG
    {
        std::cout << "build operand : {\n";
        std::cout << "\tidx : " << operandIdx << "\n";
        std::cout << "\tdata : " << data << "\n";
        std::cout << "\tsize : " << size << "\n";
        std::cout << "\ttype : " << operandType.type << "\n";
        std::cout << "\tscale : " << scale << "\n";
        std::cout << "\tzero : " << zero << "\n";
        std::cout << "\tdimensions : [ ";
        for (auto i : dims) std::cout << i << ", ";
        std::cout << "]\n}\n";
    }
#endif
    NNAPI_CHECK(ANeuralNetworksModel_addOperand_27, mNNAPIModel_, &operandType);
    if ((data != nullptr) && (size != 0U)) {
        if (useFP16) {
            halfBuffer_.emplace_back(new int16_t[size / 2]);
            FLOAT_TO_HALF(reinterpret_cast<const float *>(data), halfBuffer_.back().get(), size / 2);
            data = halfBuffer_.back().get();
        }
        if (code == ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL) {
            if (scales == nullptr) {
            }
            ANeuralNetworksSymmPerChannelQuantParams quantParam;
            quantParam.channelDim = 0;
            quantParam.scaleCount = dims[0];
            quantParam.scales = scales;
            ANeuralNetworksModel_setOperandSymmPerChannelQuantParams_29(mNNAPIModel_, operandIdx, &quantParam);
        }
        NNAPI_CHECK(ANeuralNetworksModel_setOperandValue_27, mNNAPIModel_, operandIdx, data, size);
    }
    return operandIdx;
}

ErrorCode NNAPIBackend::buildOperation(int op, const std::vector<uint32_t> &inputs, const std::vector<uint32_t> &outputs) {
    NNAPI_CHECK(ANeuralNetworksModel_addOperation_27, mNNAPIModel_, op, inputs.size(), inputs.data(), outputs.size(), outputs.data());
    return NO_ERROR;
}

ErrorCode NNAPIBackend::buildModel() {
    // TODO: add input and output
    NNAPI_CHECK(ANeuralNetworksModel_finish_27, mNNAPIModel_);
    NNAPI_CHECK(ANeuralNetworksCompilation_create_27, mNNAPIModel_, &mNNAPICompilation_);
    NNAPI_CHECK(ANeuralNetworksCompilation_finish_27, mNNAPICompilation_);
    return NO_ERROR;
}

void NNAPIBackend::invokeModel() const {
    ANeuralNetworksExecution *execution;
    NNAPI_CHECK(ANeuralNetworksExecution_create_27, mNNAPICompilation_, &execution);

    for (int i = 0; i < inputTensors_.size(); i++) {
        // const void *data = inputContentTensors_[i]->hostPtr();
        // size_t size = inputContentTensors_[i]
        const void *data = nullptr;
        size_t size = 0;
        NNAPI_CHECK(ANeuralNetworksExecution_setInput_27, execution, i, nullptr, data, size);
    }
    for (int i = 0; i < outputTensors_.size(); i++) {
        // void *data = outputContentTensors_[i]->host<void>();
        // size_t size = outputContentTensors_[i]->size();
        void *data = nullptr;
        size_t size = 0;
        NNAPI_CHECK(ANeuralNetworksExecution_setOutput_27, execution, i, nullptr, data, size);
    }

    NNAPI_CHECK(ANeuralNetworksExecution_compute_29, execution);
    ANeuralNetworksExecution_free_27(execution);
}

} // namespace mllm