#ifndef MLLM_CPUCAUSULMASK_H
#define MLLM_CPUCAUSULMASK_H

#include "Op.hpp"

namespace mllm
{   
    template <typename Dtype>
    class CPUCausalMask : public Op<Dtype> {
    public:
        CPUCausalMask(Backend *bn, bool multiThread);
        virtual ~CPUCausalMask() = default;
        virtual ErrorCode Setup(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs) override;
        virtual ErrorCode Execute(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs) override;

    private:
        bool support_multi_thread_ = false;
    };


} // namespace mllm

#endif //MLLM_CPUCAUSULMASK_H