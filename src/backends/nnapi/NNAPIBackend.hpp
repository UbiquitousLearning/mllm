#ifndef MLLM_NNAPIBACKEND_H
#define MLLM_NNAPIBACKEND_H

#include "Backend.hpp"
#include "Op.hpp"
#include "Types.hpp"
#include "NNAPIDefine.hpp"
#include "NNAPISymbol.hpp"
#include <cstdint>

namespace mllm {
class NNAPIBackend : public Backend {
public:
    NNAPIBackend(shared_ptr<MemoryManager> mm);
    ~NNAPIBackend();

    class Creator {
    public:
        virtual Op *create(OpParam op_param, Backend *bn, string name) const = 0;
    };
    void initCreatorMap() {
        map_creator_ = new std::map<OpType, NNAPIBackend::Creator *>;
    }
    bool addCreator(OpType t, Creator *c) {
        auto *map = map_creator_;
        if (map->find(t) != map->end()) {
            printf("Error: %d type has be added\n", t);
            return false;
        }
        map->insert(std::make_pair(t, c));
        return true;
    }

    virtual Op *opCreate(const OpParam &op_param, string name = "") override;

    virtual void registerOps() override;

    // NNAPI
    int bytes() const {
#ifdef USE_ARMV82
        return precision_ >= BackendConfig::PrecisionMode::Precision_Low ? 2 : 4;
#else
        return 4;
#endif
    }
    uint32_t getTensorIdx(const Tensor *t, bool dequant = false, bool isReshape = false, std::vector<uint32_t> dims = {});
    uint32_t buildScalar(int scalar);
    uint32_t buildScalar(bool scalar);
    uint32_t buildScalar(float scalar);
    uint32_t buildOperand(const void *data, size_t size, OperandCode code, std::vector<uint32_t> dims = {}, const float *scales = nullptr, int zero = 0);
    ErrorCode buildOperation(int op, const std::vector<uint32_t> &inputs, const std::vector<uint32_t> &outputs, std::string name);
    ErrorCode buildModel();
    void invokeModel() const;
    ErrorCode identifyInputsAndOutputs(std::vector<shared_ptr<Tensor>> inputs, std::vector<shared_ptr<Tensor>> outputs);

private:
    // TODO: precision config
    BackendConfig::PrecisionMode precision_ = BackendConfig::PrecisionMode::Precision_Normal;
    std::map<OpType, NNAPIBackend::Creator *> *map_creator_;
    std::vector<const Tensor *> inputTensors_, outputTensors_;
    // std::vector<std::unique_ptr<Tensor>> inputContentTensors_, outputContentTensors_;
    std::vector<std::string> opNames_;
    // tensor idx map
    std::map<const Tensor *, uint32_t> tensorIdxMap_, dequantIdxMap_;
    std::map<uint32_t, const Tensor *> dequantMap_;
    uint32_t tensorIdx_ = 0;
    // scalar idx map
    std::map<int, uint32_t> scalarIntMap_;
    std::map<bool, uint32_t> scalarBoolMap_;
    std::map<float, uint32_t> scalarFloatMap_;
    // fp16 buffer
    std::vector<std::unique_ptr<int16_t[]>> halfBuffer_;
    // NNAPI resource
    struct NNAPIDevice {
        ANeuralNetworksDevice *device;
        const char *name;
        int32_t type;
    };
    std::vector<NNAPIDevice> nnapiDevices_;
    ANeuralNetworksModel *nnapiModel_ = nullptr;
    ANeuralNetworksCompilation *nnapiCompilation_ = nullptr;
    ANeuralNetworksBurst *nnapiBurst_ = NULL;
};

} // namespace mllm

#endif // MLLM_NNAPIBACKEND_H