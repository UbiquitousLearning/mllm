#ifndef MLLM_CPUSILU_H
#define MLLM_CPUSILU_H

#include "Op.hpp"

namespace mllm
{   
    
    class CPUSiLU : public Op {
    public:
        CPUSiLU(Backend *bn, bool multiThread);
        virtual ~CPUSiLU() = default;
        virtual ErrorCode Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
        virtual ErrorCode Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;

    private:
        bool support_multi_thread_ = false;
    };


} // namespace mllm

#endif //MLLM_CPUSILU_H