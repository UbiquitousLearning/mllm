#include "NNAPIBackend.hpp"

namespace mllm {

NNAPIBackend::NNAPIBackend(shared_ptr<MemoryManager> mm) :
    Backend(mm) {
    initCreatorMap();
    registerOps();
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
    addCreator(ADD, (NNAPIBackend::Creator *)(new NNAPIAddCreator()));
    addCreator(CAUSALMASK, (NNAPIBackend::Creator *)(new NNAPICausalMaskCreator()));
    addCreator(MATMUL, (NNAPIBackend::Creator *)(new NNAPIMatmulCreator()));
    addCreator(RMSNORM, (NNAPIBackend::Creator *)(new NNAPIRMSNormCreator()));
    addCreator(ROPE, (NNAPIBackend::Creator *)(new NNAPIRoPECreator()));
    addCreator(SCALE, (NNAPIBackend::Creator *)(new NNAPIScaleCreator()));
    addCreator(SILU, (NNAPIBackend::Creator *)(new NNAPISiLUCreator()));
    addCreator(SOFTMAX, (NNAPIBackend::Creator *)(new NNAPISoftMaxCreator()));
    addCreator(LINEAR, (NNAPIBackend::Creator *)(new NNAPILinearCreator()));
}

} // namespace mllm