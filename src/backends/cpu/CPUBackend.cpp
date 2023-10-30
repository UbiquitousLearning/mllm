#include "CPUBackend.hpp"

#include "CPUAdd.hpp"
#include "CPUCausalMask.hpp"
#include "CPUMatmul.hpp"
#include "CPURMSNorm.hpp"
#include "CPURoPE.hpp"
#include "CPUScale.hpp"
#include "CPUSiLU.hpp"
#include "CPUSoftMax.hpp"
#include "CPULinear.hpp"
#include "CPUAttention.hpp"
#include "CPUEmbedding.hpp"
#include "CPUMul.hpp"
namespace mllm {
CPUBackend::CPUBackend(shared_ptr<MemoryManager>& mm) :
    Backend(mm) {
    initCreatorMap();
    registerOps();
}
// Op *CPUBackend::OpCreate(const vector<shared_ptr<Tensor>> &inputs, const vector<shared_ptr<Tensor>> &outputs,OpParam op_param)
// {
//     return map_creator_->find(optype)->second->Create(inputs, outputs, optype, this);
//     // return nullptr;
// }
Op *CPUBackend::opCreate(const OpParam &op_param, string name) {
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
    // return nullptr;
}
void CPUBackend::registerOps() {
    // ADD,
    // CAUSALMASK,
    // MATMUL,
    // RMSNORM,
    // ROPE,
    // SCALE,
    // SILU,
    // SOFTMAX

    // static CPUAddCreator _temp;
    // addCreator(ADD, &_temp);

    // static CPUMatmulCreator _temp;
    // addCreator(MATMUL, &_temp);

    addCreator(ADD, (CPUBackend::Creator *)(new CPUAddCreator()));
    addCreator(CAUSALMASK, (CPUBackend::Creator *)(new CPUCausalMaskCreator()));
    addCreator(MATMUL, (CPUBackend::Creator *)(new CPUMatmulCreator()));
    addCreator(RMSNORM, (CPUBackend::Creator *)(new CPURMSNormCreator()));
    addCreator(ROPE, (CPUBackend::Creator *)(new CPURoPECreator()));
    addCreator(SCALE, (CPUBackend::Creator *)(new CPUScaleCreator()));
    addCreator(SILU, (CPUBackend::Creator *)(new CPUSiLUCreator()));
    addCreator(SOFTMAX, (CPUBackend::Creator *)(new CPUSoftMaxCreator()));
    addCreator(LINEAR, (CPUBackend::Creator *)(new CPULinearCreator()));
    addCreator(ATTENTION, (CPUBackend::Creator *)(new CPUAttentionCreator()));
    addCreator(EMBEDDING, (CPUBackend::Creator *)(new CPUEmbeddingCreator()));
    addCreator(MUL, (CPUBackend::Creator *)(new CPUMulCreator()));
}

} // namespace mllm
