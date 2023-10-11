#ifndef MLLM_CPULINEAR_H
#define MLLM_CPULINEAR_H

#include "Op.hpp"
#include "CPUBackend.hpp"
#include "compute/StrassenMatmul.hpp"

namespace mllm {

class Tensor;
class CPULinear final : public Op {
public:
    CPULinear(Backend *bn, string opName, int in_features, int out_features, bool bias, bool multiThread);
    virtual ~CPULinear() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode reshapeOutputs(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs);

    virtual ErrorCode load(ParamLoader &loader) override;

private:
    int in_features_;
    int out_features_;
    bool support_bias_;
    bool support_multi_thread_ = false;
    Tensor weight_;
    Tensor bias_;
};

class CPULinearCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        int in_features = op_param["in_features"];
        int out_features = op_param["out_features"];
        int bias = op_param["bias"];
        return new CPULinear(bn, name, in_features, out_features, (bool)bias, false);
    }
};

} // namespace mllm

#endif // MLLM_CPULINEAR_H