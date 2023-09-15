#ifndef MLLM_CPUMATMUL_H
#define MLLM_CPUMATMUL_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm
{   
    
class Tensor;
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

    class CPUMatmulCreator : public CPUBackend::Creator {
    public:
        // virtual Op *Create(const vector<shared_ptr<Tensor>>& inputs, const vector<shared_ptr<Tensor>>& outputs,
        //                                 OpType optype, Backend* backend) const  {
        //     return new CPUMatmul(backend, false, false, false, false);
        // }
        virtual Op *Create(OpType optype, Backend* backend) const  {
            return new CPUMatmul(backend, false, false, false, false);
        }

    };





} // namespace mllm

#endif //MLLM_CPUMATMUL_H