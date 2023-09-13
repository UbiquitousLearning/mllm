#ifndef MLLM_CPUSCALE_H
#define MLLM_CPUSCALE_H

#include "Op.hpp"

namespace mllm
{   
    template <typename Dtype>
    class CPUScale : public Op<Dtype> {
    public:
        CPUScale(Backend *bn, bool multiThread);
        virtual ~CPUScale() = default;
        virtual ErrorCode Setup(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs) override;
        virtual ErrorCode Execute(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs) override;

    private:
        bool support_multi_thread_ = false;
    };


} // namespace mllm

#endif //MLLM_CPUSCALE_H