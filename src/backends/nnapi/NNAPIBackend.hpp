#ifndef MLLM_NNAPIBACKEND_H
#define MLLM_NNAPIBACKEND_H

#include "Backend.hpp"
#include "Op.hpp"
#include "Types.hpp"
#include "NNAPIDefine.hpp"
#include "NNAPISymbol.hpp"

namespace mllm {
class NNAPIBackend : public Backend {
public:
    NNAPIBackend(shared_ptr<MemoryManager> mm);
    ~NNAPIBackend();

    class Creator {
    public:
        // virtual Op* Create(const vector<shared_ptr<Tensor>>& inputs, const vector<shared_ptr<Tensor>>& outputs,
        //                             OpParam op_param, Backend* backend) const = 0;
        virtual Op *create(OpParam op_param, Backend *bn) const = 0;
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

    // virtual Op* OpCreate(const vector<shared_ptr<Tensor>>& inputs, const vector<shared_ptr<Tensor>>& outputs,
    //                             OpParam op_param) override;

    virtual Op *opCreate(const OpParam &op_param) override;

    virtual void registerOps() override;

private:
    std::map<OpType, NNAPIBackend::Creator *> *map_creator_;
    // NNAPI resource
    struct NNAPIDevice {
        ANeuralNetworksDevice *device;
        const char *name;
        int32_t type;
    };
    std::vector<NNAPIDevice> mNNAPIDevices_;
    ANeuralNetworksModel *mNNAPIModel_ = nullptr;
    ANeuralNetworksCompilation *mNNAPICompilation_ = nullptr;
    ANeuralNetworksBurst *mNNAPIBurst_ = NULL;
};

} // namespace mllm

#endif // MLLM_NNAPIBACKEND_H