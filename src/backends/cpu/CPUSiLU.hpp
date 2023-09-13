#ifndef MLLM_CPUSILU_H
#define MLLM_CPUSILU_H

#include "Op.hpp"

namespace mllm
{   
    template <typename Dtype>
    class CPUSiLU : public Op<Dtype> {
    public:
        CPUSiLU(Backend *bn, bool multiThread);
        virtual ~CPUSiLU() = default;
        virtual ErrorCode Setup(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs) override;
        virtual ErrorCode Execute(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs) override;

    private:
        bool support_multi_thread_ = false;
    };


} // namespace mllm

#endif //MLLM_CPUSILU_H