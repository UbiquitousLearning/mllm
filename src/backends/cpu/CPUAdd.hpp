#ifndef MLLM_CPUADD_H
#define MLLM_CPUADD_H

#include "Op.hpp"

namespace mllm
{   
    
    class CPUAdd : public Op {
    public:
        CPUAdd(Backend *bn, bool multiThread);
        virtual ~CPUAdd() = default;
        virtual ErrorCode Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
        virtual ErrorCode Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;

    private:
        bool support_multi_thread_ = false;
    };


} // namespace mllm

#endif //MLLM_CPUADD_H