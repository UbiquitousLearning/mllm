#ifndef MLLM_CPUCAUSULMASK_H
#define MLLM_CPUCAUSULMASK_H

#include "Op.hpp"

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


} // namespace mllm

#endif //MLLM_CPUCAUSULMASK_H