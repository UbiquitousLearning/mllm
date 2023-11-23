#ifndef MLLM_QNNBACKEND_H
#define MLLM_QNNBACKEND_H

#include "MemoryManager.hpp"
#include "NetParameter.hpp"
#include <memory>
using std::shared_ptr;

namespace mllm {
class Op;

class Tensor;
class Backend;
class QNNBackend : public Backend {
public:
    QNNBackend(){};
    Backend(shared_ptr<MemoryManager>& mm) :
        mem_manager_(mm) {
        // nothing to do
    }
    virtual ~Backend() = default;

    // Init QNN Backend context
    void init() override; // TODO: Config

    void release() override;

    // void alloc(void **ptr, size_t size,size_t alignment) {
    //     mem_manager_->alloc(ptr, size, alignment);
    // }

    // void free(void *ptr) {
    //     mem_manager_->free(ptr);
    // }

    
    /**
     * @brief create execution for op with input and output tensors.
     * @param inputs    input tensors.
     * @param outputs   output tensors.
     * @param op        given op.
     * @return created execution if op is supported, nullptr otherwise.
     */
    // virtual Op* OpCreate(const vector<shared_ptr<Tensor>>& inputs, const vector<shared_ptr<Tensor>>& outputs,
    //                             OpParam op_param) = 0;
    virtual Op *opCreate(const OpParam &op_param, string name="") = 0;
    virtual void registerOps() = 0;
    // virtual void* OpCreater(OpParam op_param);
private:
    //
    shared_ptr<MemoryManager> mem_manager_;
    // unordered_map<OpType, Op*(*)(Backend*)> op_map_;


    


};

} // namespace mllm

#endif // MLLM_QNNBACKEND_H