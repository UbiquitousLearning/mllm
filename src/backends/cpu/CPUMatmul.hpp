#ifndef MLLM_CPUMATMUL_H
#define MLLM_CPUMATMUL_H

#include "Op.hpp"
#include "CPUBackend.hpp"
#include "compute/Matmul.hpp"

namespace mllm {

class Tensor;
class CPUMatmul final : public Op {
public:
    CPUMatmul(Backend *bn, string opName, bool transpose0, bool transpose1, bool multiThread);
    virtual ~CPUMatmul() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    virtual ErrorCode load(ParamLoader &loader) override;

private:
    bool transpose0_;
    bool transpose1_;
    bool support_multi_thread_ = false;
};

class CPUMatmulCreator : public CPUBackend::Creator {
public:
    // virtual Op *Create(const vector<shared_ptr<Tensor>>& inputs, const vector<shared_ptr<Tensor>>& outputs,
    //                                 OpParam op_param, Backend* backend) const  {
    //     return new CPUMatmul(backend, false, false, false, false);
    // }
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new CPUMatmul(bn, name, false, false, false);
    }
};

} // namespace mllm

#endif // MLLM_CPUMATMUL_H