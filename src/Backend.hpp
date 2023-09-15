#ifndef MLLM_BACKEND_H
#define MLLM_BACKEND_H

#include "MemoryManager.hpp"
#include "Op.hpp"
namespace mllm
{
class Op;

class Tensor;
class Backend;
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

        // unordered_map<OpType, Op*(*)(Backend*)> OpMap(){
        //     return op_map_;
        // }

        // void Execute() {

        // }

        // bool CPUTensorConvert(shared_ptr<Tensor> src_tensor, shared_ptr<Tensor> dst_tensor, int type_); //NCHW --> NHWC ..., TODO type_:enum

            /**
         * @brief create execution for op with input and output tensors.
         * @param inputs    input tensors.
         * @param outputs   output tensors.
         * @param op        given op.
         * @return created execution if op is supported, nullptr otherwise.
         */
        // virtual Op* OpCreate(const vector<shared_ptr<Tensor>>& inputs, const vector<shared_ptr<Tensor>>& outputs,
        //                             OpType optype) = 0;
        virtual Op* OpCreate(OpType optype) = 0;
        virtual void registerOps() = 0;
        // virtual void* OpCreater(OpType optype);
    private:
        //
        shared_ptr<MemoryManager> mem_manager_;
        // unordered_map<OpType, Op*(*)(Backend*)> op_map_;

    };
    
} // namespace mllm



#endif //MLLM_BACKEND_H