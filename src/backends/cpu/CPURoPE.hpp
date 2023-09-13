#ifndef MLLM_CPUROPE_H
#define MLLM_CPUROPE_H

#include "Op.hpp"

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


} // namespace mllm

#endif //MLLM_CPUROPE_H