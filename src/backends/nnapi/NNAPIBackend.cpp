#include "NNAPIBackend.hpp"
#include "NNAPISymbol.hpp"
#include "Types.hpp"
#include <cstdint>
#include <iostream>
#include <vector>
#include "NNAPIAdd.hpp"
#include "NNAPIMatmul.hpp"

// TODO: float <--> half convert for armv82
#define FLOAT_TO_HALF(...)
#define HALF_TO_FLOAT(...)

#define NNAPI_CHECK(func, ...)                                    \
    do {                                                          \
        const auto _status = (func(__VA_ARGS__));                 \
        if (_status != ANEURALNETWORKS_NO_ERROR) {                \
            const auto ENUM_TO_STR = NNAPIEnumToString(_status);  \
            std::cout << "NNAPI error : " << ENUM_TO_STR << "\n"; \
            exit(0);                                              \
        }                                                         \
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

static uint16_t fp32to16(float val) {
    uint32_t x = *((uint32_t *)&val);
    uint16_t h = ((x >> 16) & 0x8000) | ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((x >> 13) & 0x03ff);
    return h;
}

NNAPIBackend::NNAPIBackend(shared_ptr<MemoryManager> mm) :
    Backend(mm) {
    loadNNAPISymbol();
    initCreatorMap();
    registerOps();

    if (nnapiModel_ == nullptr) {
        NNAPI_CHECK(ANeuralNetworksModel_create_27, &nnapiModel_);
    }
    if (nnapiDevices_.empty()) {
        uint32_t numDevices = 0;
        NNAPI_CHECK(ANeuralNetworks_getDeviceCount_29, &numDevices);
        nnapiDevices_.resize(numDevices);
#ifdef DEBUG
        std::cout << "NNAPI numDevices :" << numDevices << "\n";
#endif
        for (int i = 0; i < numDevices; i++) {
            NNAPI_CHECK(ANeuralNetworks_getDevice_29, i, &nnapiDevices_[i].device);
            NNAPI_CHECK(ANeuralNetworksDevice_getName_29, nnapiDevices_[i].device, &nnapiDevices_[i].name);
            NNAPI_CHECK(ANeuralNetworksDevice_getType_29, nnapiDevices_[i].device, &nnapiDevices_[i].type);
        }
    }
}

NNAPIBackend::~NNAPIBackend() {
    ANeuralNetworksCompilation_free_27(nnapiCompilation_);
    ANeuralNetworksModel_free_27(nnapiModel_);
}

// don't pass name to op
Op *NNAPIBackend::opCreate(const OpParam &op_param, string name) {
    OpType optype = OpType(op_param.find("type")->second);
    auto *map = map_creator_;
    auto iter = map->find(optype);
    if (iter == map->end()) {
        printf("Don't support type \n");
        return nullptr;
    }
    Op *exe = nullptr;
    exe = iter->second->create(op_param, this, name);
    return exe;
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
    addCreator(ADD, (NNAPIBackend::Creator *)(new NNAPIAddCreator()));
    // addCreator(CAUSALMASK, (NNAPIBackend::Creator *)(new NNAPICausalMaskCreator()));
    addCreator(MATMUL, (NNAPIBackend::Creator *)(new NNAPIMatmulCreator()));
    // addCreator(RMSNORM, (NNAPIBackend::Creator *)(new NNAPIRMSNormCreator()));
    // addCreator(ROPE, (NNAPIBackend::Creator *)(new NNAPIRoPECreator()));
    // addCreator(SCALE, (NNAPIBackend::Creator *)(new NNAPIScaleCreator()));
    // addCreator(SILU, (NNAPIBackend::Creator *)(new NNAPISiLUCreator()));
    // addCreator(SOFTMAX, (NNAPIBackend::Creator *)(new NNAPISoftMaxCreator()));
    // addCreator(LINEAR, (NNAPIBackend::Creator *)(new NNAPILinearCreator()));
}

uint32_t NNAPIBackend::getTensorIdx(const Tensor *t, bool dequant) {
    // for input and output tensor, save them in inputTensors_ and outputTensors_
    // TODO: add INPUT and OUTPUT description in tensor for efficiency
    auto isInput = std::find(inputTensors_.begin(), inputTensors_.end(), t) != inputTensors_.end();
    auto isOutput = std::find(outputTensors_.begin(), outputTensors_.end(), t) != outputTensors_.end();

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
    // TODO: ANEURALNETWORKS_TENSOR_INT32 and ANEURALNETWORKS_TENSOR_QUANT8_ASYMM
    auto code = ANEURALNETWORKS_TENSOR_FLOAT32;
    uint32_t idx = -1;
    // TODO: CONSTANT operand shoul have no size and nullptr data
    if (isInput || isOutput) {
        // for input and output tensor, don't set data when add operand for nnapi model
        idx = buildOperand(nullptr, t->size(), code, udims, &scale, zero);
    } else {
        idx = buildOperand(t->hostPtr<void>(), t->size(), code, udims, &scale, zero);
    }
    tensorIdxMap_.insert(std::make_pair(t, idx));
    return idx;
}

uint32_t NNAPIBackend::buildScalar(int scalar) {
    auto iter = scalarIntMap_.find(scalar);
    if (iter != scalarIntMap_.end()) {
        return iter->second;
    }
    auto scalarIdx = buildOperand(&scalar, 4, ANEURALNETWORKS_INT32);
    scalarIntMap_.insert(std::make_pair(scalar, scalarIdx));
    return scalarIdx;
}

uint32_t NNAPIBackend::buildScalar(bool scalar) {
    auto iter = scalarBoolMap_.find(scalar);
    if (iter != scalarBoolMap_.end()) {
        return iter->second;
    }
    uint8_t value = static_cast<uint8_t>(scalar);
    auto scalarIdx = buildOperand(&value, 1, ANEURALNETWORKS_BOOL);
    scalarBoolMap_.insert(std::make_pair(scalar, scalarIdx));
    return scalarIdx;
}

uint32_t NNAPIBackend::buildScalar(float scalar) {
    auto iter = scalarFloatMap_.find(scalar);
    if (iter != scalarFloatMap_.end()) {
        return iter->second;
    }
    uint32_t scalarIdx = -1;
    if (bytes() == 2) {
        uint16_t value = fp32to16(scalar);
        scalarIdx = buildOperand(&value, 2, ANEURALNETWORKS_FLOAT16);
    } else {
        scalarIdx = buildOperand(&scalar, 4, ANEURALNETWORKS_FLOAT32);
    }
    scalarFloatMap_.insert(std::make_pair(scalar, scalarIdx));
    return scalarIdx;
}

uint32_t NNAPIBackend::buildOperand(const void *data, size_t size, OperandCode code, std::vector<uint32_t> dims, const float *scales, int zero) {
    // TODO: fp16 and quant8 support
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
    NNAPI_CHECK(ANeuralNetworksModel_addOperand_27, nnapiModel_, &operandType);
    if ((data != nullptr) && (size != 0U)) {
        // TODO: fp16 and quant8 support
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
            ANeuralNetworksModel_setOperandSymmPerChannelQuantParams_29(nnapiModel_, operandIdx, &quantParam);
        }
        NNAPI_CHECK(ANeuralNetworksModel_setOperandValue_27, nnapiModel_, operandIdx, data, size);
#ifdef DEBUG
        {
            std::cout << "set operand value : {\n";
            std::cout << "\tidx : " << operandIdx << "\n";
            std::cout << "\tdata : " << data << "\n";
            std::cout << "\tsize : " << size << "\n";
            if (code == ANEURALNETWORKS_TENSOR_FLOAT32) {
                float *data_float = static_cast<float *>(const_cast<void *>(data));
                std::cout << "data_float: " << data_float[0] << std::endl;
            }
            std::cout << "}\n";
        }
#endif
    }
    return operandIdx;
}

ErrorCode NNAPIBackend::buildOperation(int op, const std::vector<uint32_t> &inputs, const std::vector<uint32_t> &outputs, std::string name) {
#ifdef DEBUG
    {
        std::cout << "build operation : {\n";
        std::cout << "\tname : " << name << "\n";
        std::cout << "\ttype : " << op << "\n";
        std::cout << "\tinputs : [ ";
        for (auto i : inputs) std::cout << i << ", ";
        std::cout << "]\n\toutputs : [ ";
        for (auto i : outputs) std::cout << i << ", ";
        std::cout << "]\n}\n";
    }
#endif
    opNames_.push_back(name);
    NNAPI_CHECK(ANeuralNetworksModel_addOperation_27, nnapiModel_, op, inputs.size(), inputs.data(), outputs.size(), outputs.data());
    return NO_ERROR;
}

ErrorCode NNAPIBackend::buildModel() {
    // set input and output of model
    std::vector<uint32_t> inputOperands(inputTensors_.size());
    std::vector<uint32_t> outputOperands(outputTensors_.size());
    for (int i = 0; i < inputTensors_.size(); i++) {
        inputOperands[i] = getTensorIdx(inputTensors_[i]);
    }
    for (int i = 0; i < outputTensors_.size(); i++) {
        const auto *output = outputTensors_[i];
        outputOperands[i] = getTensorIdx(outputTensors_[i]);
    }
#ifdef DEBUG
    {
        std::cout << "set model's inputs & outputs : {\n";
        std::cout << "\tinputs : [ ";
        for (auto i : inputOperands) std::cout << i << ", ";
        std::cout << "]\n\toutputs : [ ";
        for (auto i : outputOperands) std::cout << i << ", ";
        std::cout << "]\n}\n";
    }
#endif
    NNAPI_CHECK(ANeuralNetworksModel_identifyInputsAndOutputs_27,
                nnapiModel_,
                inputOperands.size(),
                inputOperands.data(),
                outputOperands.size(),
                outputOperands.data());
    NNAPI_CHECK(ANeuralNetworksModel_finish_27, nnapiModel_);

    std::unique_ptr<bool[]> supports(new bool[opNames_.size()]);
    int selectDeviceIdx = -1;
    for (int i = 0; i < nnapiDevices_.size(); i++) {
        auto *device = nnapiDevices_[i].device;
        const auto *name = nnapiDevices_[i].name;
        auto type = nnapiDevices_[i].type;
        NNAPI_CHECK(ANeuralNetworksModel_getSupportedOperationsForDevices_29, nnapiModel_, &device, 1, supports.get());
#ifdef DEBUG
        std::cout << "device [" << i << " : " << name << "] supportOps = {\n";
#endif
        bool allsupport = true;
        for (int i = 0; i < opNames_.size(); i++) {
            allsupport &= supports[i];
#ifdef DEBUG
            std::cout << "\t" << opNames_[i] << " : " << supports[i] << "\n";
#endif
        }
#ifdef DEBUG
        std::cout << "}\n";
#endif
        if (allsupport) {
            selectDeviceIdx = i;
#ifdef DEBUG
            std::cout << "[NNAPI] using device [" << i << " : " << name << " : " << type << "].\n";
#endif
            break;
        }
    }
    std::cout << "[NNAPI] using device [" << selectDeviceIdx << " : " << nnapiDevices_[selectDeviceIdx].name << "].\n";
    NNAPI_CHECK(ANeuralNetworksCompilation_createForDevices_29, nnapiModel_, &nnapiDevices_[selectDeviceIdx].device, 1, &nnapiCompilation_);
    NNAPI_CHECK(ANeuralNetworksCompilation_setPreference_27, nnapiCompilation_, ANEURALNETWORKS_PREFER_SUSTAINED_SPEED);
    NNAPI_CHECK(ANeuralNetworksCompilation_finish_27, nnapiCompilation_);
    NNAPI_CHECK(ANeuralNetworksBurst_create_29, nnapiCompilation_, &nnapiBurst_);
    return NO_ERROR;
}

void NNAPIBackend::invokeModel() const {
#ifdef DEBUG
    std::cout << "[NNAPI] invoke model.\n";
#endif
    ANeuralNetworksExecution *execution;
    NNAPI_CHECK(ANeuralNetworksExecution_create_27, nnapiCompilation_, &execution);

    for (int i = 0; i < inputTensors_.size(); i++) {
        const void *data = inputTensors_[i]->hostPtr<void>();
        size_t size = inputTensors_[i]->size();
        NNAPI_CHECK(ANeuralNetworksExecution_setInput_27, execution, i, nullptr, data, size);
    }

    for (int i = 0; i < outputTensors_.size(); i++) {
        void *data = outputTensors_[i]->hostPtr<void>();
        size_t size = outputTensors_[i]->size();
        NNAPI_CHECK(ANeuralNetworksExecution_setOutput_27, execution, i, nullptr, data, size);
    }

    NNAPI_CHECK(ANeuralNetworksExecution_compute_29, execution);
    ANeuralNetworksExecution_free_27(execution);
}

ErrorCode NNAPIBackend::identifyInputsAndOutputs(std::vector<shared_ptr<Tensor>> inputs, std::vector<shared_ptr<Tensor>> outputs) {
    // in case of inputs passed from graph have two same value, inputTensors_ should dynamically resize
    // TODO: better solution(tensor description)
    // inputTensors_.resize(inputs.size());
    outputTensors_.resize(outputs.size());
    for (int i = 0; i < inputs.size(); i++) {
        // inputTensors should not have same value
        if (i > 0 && inputTensors_[i - 1] == inputs[i].get()) {
            continue;
        }
        inputTensors_.emplace_back(inputs[i].get());
    }
    for (int i = 0; i < outputs.size(); i++) {
        outputTensors_[i] = outputs[i].get();
    }
    return NO_ERROR;
}

} // namespace mllm