#include "CPUBackend.hpp"

#include "CPUAdd.hpp"
#include "CPUCausalMask.hpp"
#include "CPUMatmul.hpp"
#include "CPURMSNorm.hpp"
#include "CPURoPE.hpp"
#include "CPUScale.hpp"
#include "CPUSiLU.hpp"
#include "CPUSoftMax.hpp"
namespace mllm {
CPUBackend::CPUBackend(shared_ptr<MemoryManager> mm) :
    Backend(mm) {
    initCreatorMap();
    RegisterOps();
}
// Op *CPUBackend::OpCreate(const vector<shared_ptr<Tensor>> &inputs, const vector<shared_ptr<Tensor>> &outputs,OpParam op_param)
// {
//     return map_creator_->find(optype)->second->Create(inputs, outputs, optype, this);
//     // return nullptr;
// }
Op *CPUBackend::OpCreate(const OpParam &op_param) {
    OpType optype = OpType(op_param.find("type")->second);
    auto map = map_creator_;
    auto iter = map->find(optype);
    if (iter == map->end()) {
        printf("Don't support type \n");
        return nullptr;
    }
    Op *exe = nullptr;
    if (exe == nullptr) {
        exe = iter->second->Create(op_param, this);
    }
    return exe;
    // return nullptr;
}
void CPUBackend::RegisterOps() {
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

    addCreator(Add, (CPUBackend::Creator *)(new CPUAddCreator()));
    addCreator(CausalMask, (CPUBackend::Creator *)(new CPUCausalMaskCreator()));
    addCreator(Matmul, (CPUBackend::Creator *)(new CPUMatmulCreator()));
    addCreator(RMSNorm, (CPUBackend::Creator *)(new CPURMSNormCreator()));
    addCreator(RoPE, (CPUBackend::Creator *)(new CPURoPECreator()));
    addCreator(Scale, (CPUBackend::Creator *)(new CPUScaleCreator()));
    addCreator(Silu, (CPUBackend::Creator *)(new CPUSiLUCreator()));
    addCreator(SoftMax, (CPUBackend::Creator *)(new CPUSoftMaxCreator()));
}

} // namespace mllm
