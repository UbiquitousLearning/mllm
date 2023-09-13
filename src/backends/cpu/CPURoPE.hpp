#ifndef MLLM_CPUROPE_H
#define MLLM_CPUROPE_H

#include "Op.hpp"

namespace mllm
{   
    template <typename Dtype>
    class CPURoPE : public Op<Dtype> {
    public:
        CPURoPE(Backend *bn, bool multiThread);
        virtual ~CPURoPE() = default;
        virtual ErrorCode Setup(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs) override;
        virtual ErrorCode Execute(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs) override;

    private:
        bool support_multi_thread_ = false;
    };


} // namespace mllm

#endif //MLLM_CPUROPE_H