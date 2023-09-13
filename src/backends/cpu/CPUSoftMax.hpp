#ifndef MLLM_CPUSOFTMAX_H
#define MLLM_CPUSOFTMAX_H

#include "Op.hpp"

namespace mllm
{   
    
    class CPUSoftMax : public Op {
    public:
        CPUSoftMax(Backend *bn, bool multiThread);
        virtual ~CPUSoftMax() = default;
        virtual ErrorCode Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
        virtual ErrorCode Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;

    private:
        bool support_multi_thread_ = false;
    };


} // namespace mllm

#endif //MLLM_CPUSOFTMAX_H