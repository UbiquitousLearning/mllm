#ifndef MLLM_CPUROPE_H
#define MLLM_CPUROPE_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm
{   
    
    class CPURoPE : public Op {
    public:
        CPURoPE(Backend *bn, bool multiThread);
        virtual ~CPURoPE() = default;
        virtual ErrorCode Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
        virtual ErrorCode Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;

    private:
        bool support_multi_thread_ = false;
    };


    class CPURoPECreator : public CPUBackend::Creator {
    public:
        virtual Op *Create(OpType optype, Backend* bn) const  {
            return new CPURoPE(bn, false);
        }
    };
} // namespace mllm

#endif //MLLM_CPUROPE_H