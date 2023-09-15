#include "CPUBackend.hpp"

#include "CPUAdd.hpp"
#include "CPUCausalMask.hpp"
#include "CPUMatmul.hpp"
#include "CPURMSNorm.hpp"
#include "CPURoPE.hpp"
#include "CPUScale.hpp"
#include "CPUSiLU.hpp"
#include "CPUSoftMax.hpp"
namespace mllm
{
    CPUBackend::CPUBackend(shared_ptr<MemoryManager> mm): Backend(mm)
    {
        initCreatorMap();
    }
    // Op *CPUBackend::OpCreate(const vector<shared_ptr<Tensor>> &inputs, const vector<shared_ptr<Tensor>> &outputs,OpType optype)
    // {
    //     return map_creator_->find(optype)->second->Create(inputs, outputs, optype, this);
    //     // return nullptr;
    // }
    Op *CPUBackend::OpCreate(OpType optype)
    {

        auto map  = map_creator_;
        auto iter = map->find(optype);
        if (iter == map->end()) {
            printf("Don't support type \n");
            return nullptr;
        }
        Op* exe = nullptr;
        if (exe == nullptr) {
            exe = iter->second->Create(optype, this);
        }
        return exe;
        // return nullptr;
    }
    void CPUBackend::registerOps()
    {
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
        
        addCreator(ADD, (CPUBackend::Creator*)(new CPUAddCreator()));
        addCreator(CAUSALMASK, (CPUBackend::Creator*)(new CPUCausalMaskCreator()));
        addCreator(MATMUL, (CPUBackend::Creator*)(new CPUMatmulCreator()));
        addCreator(RMSNORM, (CPUBackend::Creator*)(new CPURMSNormCreator()));
        addCreator(ROPE, (CPUBackend::Creator*)(new CPURoPECreator()));
        addCreator(SCALE, (CPUBackend::Creator*)(new CPUScaleCreator()));
        addCreator(SILU, (CPUBackend::Creator*)(new CPUSiLUCreator()));
        addCreator(SOFTMAX, (CPUBackend::Creator*)(new CPUSoftMaxCreator()));
        


    }

} // namespace mllm
