
#ifndef MLLM_CPUSPARSEIDLINEAR_H
#define MLLM_CPUSPARSEIDLINEAR_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUSparseIdLinear final : public Op {
public:
    CPUSparseIdLinear(Backend *bn, string opName, int in_dim, int out_dim, int threadCount);
    ~CPUSparseIdLinear() override = default;
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode load(AbstructLoader &loader) override;
    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int in_dim_;
    int out_dim_;
    int thread_count = 4;
    Tensor weight_; // weight of shape [out_dim_, in_dim_].  dst = x * weight^T  (for contiguously access memory)
};

class CPUSparseIdLinearCreator : public CPUBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        int in_dim_ = (int)op_param["in_dim_"];
        int out_dim_ = (int)op_param["out_dim_"];
        return new CPUSparseIdLinear(bn, name, in_dim_, out_dim_, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUSPARSEIDLINEAR_H
