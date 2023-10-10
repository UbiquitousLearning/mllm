#ifndef MLLM_BACKEND_H
#define MLLM_BACKEND_H

#include "MemoryManager.hpp"
#include "NetParameter.hpp"
namespace mllm {
class Op;

class Tensor;
class Backend;
class Backend {
public:
    Backend(){};
    Backend(shared_ptr<MemoryManager>& mm) :
        mem_manager_(mm) {
        // nothing to do
    }
    virtual ~Backend() = default;

    // bool CheckSupport(shared_ptr<Op> op) {
    //     // return OPMap.contains(op->type);
    //     return true;
    // }

    void init(){}; // TODO: Config

    void release(){};

    void alloc(void **ptr, size_t size) {
        mem_manager_->alloc(ptr, size);
    }

    void free(void *ptr) {
        mem_manager_->free(ptr);
    }

    // unordered_map<OpType, Op*(*)(Backend*)> OpMap(){
    //     return op_map_;
    // }

    // void execute() {

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
    //                             OpParam op_param) = 0;
    virtual Op *opCreate(const OpParam &op_param) = 0;
    virtual void registerOps() = 0;
    // virtual void* OpCreater(OpParam op_param);
private:
    //
    shared_ptr<MemoryManager> mem_manager_;
    // unordered_map<OpType, Op*(*)(Backend*)> op_map_;
};

} // namespace mllm

#endif // MLLM_BACKEND_H