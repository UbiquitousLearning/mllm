#include "NNAPIBackend.hpp"

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

} // namespace mllm