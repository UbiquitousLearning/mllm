#ifndef MLLM_CPURMSNORM_H
#define MLLM_CPURMSNORM_H

#include "Op.hpp"

namespace mllm
{   
    template <typename Dtype>
    class CPURMSNorm : public Op<Dtype> {
    public:
        CPURMSNorm(Backend *bn, bool multiThread);
        virtual ~CPURMSNorm() = default;
        virtual ErrorCode Setup(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs) override;
        virtual ErrorCode Execute(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs) override;

    private:
        bool support_multi_thread_ = false;
    };


} // namespace mllm

#endif //MLLM_CPURMSNORM_H