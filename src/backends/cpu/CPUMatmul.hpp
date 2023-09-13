#ifndef MLLM_CPUMATMUL_H
#define MLLM_CPUMATMUL_H

#include "Op.hpp"

namespace mllm
{   
    
    class CPUMatmul : public Op {
    public:
        CPUMatmul(Backend *bn, bool transposeA, bool transposeB, bool transposeC, bool multiThread);
        virtual ~CPUMatmul() = default;
        virtual ErrorCode Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
        virtual ErrorCode Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;

    private:        
        bool transposeA_;
        bool transposeB_;
        bool transposeC_;
        bool support_multi_thread_ = false;
    };


} // namespace mllm

#endif //MLLM_CPUMATMUL_H