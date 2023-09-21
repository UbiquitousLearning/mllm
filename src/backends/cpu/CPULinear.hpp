#ifndef MLLM_CPULINEAR_H
#define MLLM_CPULINEAR_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class Tensor;
class CPULinear : public Op {
public:
    CPULinear(Backend *bn, int in_features, int out_features, bool bias, bool multiThread);
    virtual ~CPULinear() = default;
    virtual ErrorCode Reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
    virtual ErrorCode Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
    virtual ErrorCode Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;

    virtual ErrorCode Load(ParamLoader &loader) override;

private:
    bool support_multi_thread_ = false;
};

class CPULinearCreator : public CPUBackend::Creator {
public:
    virtual Op *Create(OpParam op_param, Backend *bn) const {
        int in_features = 1;
        int out_features = 1;
        return new CPULinear(bn, in_features, out_features, false, false);
    }
};

} // namespace mllm

#endif // MLLM_CPULINEAR_H