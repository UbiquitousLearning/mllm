#ifndef MLLM_CPULINEAR_H
#define MLLM_CPULINEAR_H

#include "Op.hpp"
#include "../CPUBackend.hpp"
#include "../compute/Matmul.hpp"

namespace mllm {

class Tensor;
class CPULinear final : public Op {
public:
    CPULinear(Backend *bn, string opName, int in_features, int out_features, bool bias, int threadCount);
    virtual ~CPULinear() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    Tensor &weight() {
        return weight_;
    }
    Tensor &bias() {
        return bias_;
    }

private:
    int in_features_;
    int out_features_;
    bool support_bias_;
    int thread_count = 4;
    Tensor weight_;
    Tensor bias_;
};

class CPULinearCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int in_features = op_param["in_features"];
        int out_features = op_param["out_features"];
        int bias = op_param["bias"];
        return new CPULinear(bn, name, in_features, out_features, (bool)bias, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPULINEAR_H