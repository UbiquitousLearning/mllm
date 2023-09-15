#ifndef MLLM_CPUCAUSULMASK_H
#define MLLM_CPUCAUSULMASK_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm
{   
    
    class CPUCausalMask : public Op {
    public:
        CPUCausalMask(Backend *bn, bool multiThread);
        virtual ~CPUCausalMask() = default;
        virtual ErrorCode Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
        virtual ErrorCode Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;

    private:
        bool support_multi_thread_ = false;
    };


    class CPUCausalMaskCreator : public CPUBackend::Creator {
    public:
        virtual Op *Create(OpType optype, Backend* bn) const  {
            return new CPUCausalMask(bn, false);
        }

    };
} // namespace mllm

#endif //MLLM_CPUCAUSULMASK_H