#include "CPUBackend.hpp"
#include "CPUMatmul.hpp"
namespace mllm
{
    CPUBackend::CPUBackend(shared_ptr<MemoryManager> mm): Backend(mm)
    {
        initCreatorMap();
        // REGISTER_CPU_OP_CREATOR(CPUMatmulCreator, MATMUL);
        // addCreator(MATMUL, CPUMatmulCreator);

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
        
            CPUBackend::initCreatorMap();

            static CPUMatmulCreator _temp;
            CPUBackend::addCreator(MATMUL, &_temp);

        //     ___CPUMatmulCreator__MATMUL__();

    }
    std::map<OpType, CPUBackend::Creator *> *CPUBackend::map_creator_ = nullptr;

    // void registerCPUOps(){
    //     CPUBackend::addCreator(MATMUL, )
    // }

} // namespace mllm
