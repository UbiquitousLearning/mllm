#ifndef MLLM_BACKEND_H
#define MLLM_BACKEND_H

#include "MemoryManager.hpp"
namespace mllm
{
    class Backend {
    public:
        Backend(){};
        Backend(shared_ptr<MemoryManager> mm) : mem_manager_(mm) {
            // nothing to do
        }
        virtual ~Backend() = default;

        // bool CheckSupport(shared_ptr<Op> op) {
        //     // return OPMap.contains(op->type);
        //     return true;
        // }
        
        // bool CheckSupport(shared_ptr<Op> op) {
        //     // return OPMap.contains(op->type);
        //     return true;
        // }
        void Init(); //TODO: Config

        void Release();

        void Alloc(void** ptr, size_t size){
            mem_manager_->Alloc(ptr, size);
        }

        void Free(void** ptr){
            mem_manager_->Free(ptr);
        }

        // void Execute() {

        // }

        // bool CPUTensorConvert(shared_ptr<Tensor> src_tensor, shared_ptr<Tensor> dst_tensor, int type_); //NCHW --> NHWC ..., TODO type_:enum
    private:
        //
        shared_ptr<MemoryManager> mem_manager_;

    };
    
} // namespace mllm



#endif //MLLM_BACKEND_H