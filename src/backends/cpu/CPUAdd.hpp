#ifndef MLLM_CPUADD_H
#define MLLM_CPUADD_H

#include "Op.hpp"

namespace mllm
{   
    template <typename Dtype>
    class CPUAdd : public Op<Dtype> {
    public:
        CPUAdd(Backend *bn, bool multiThread);
        virtual ~CPUAdd() = default;
        virtual ErrorCode Setup(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs) override;
        virtual ErrorCode Execute(vector<shared_ptr<Tensor<Dtype>>> &inputs, vector<shared_ptr<Tensor<Dtype>>> &outputs) override;

    private:
        bool support_multi_thread_ = false;
    };


} // namespace mllm

#endif //MLLM_CPUADD_H