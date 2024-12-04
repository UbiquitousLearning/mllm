
#ifndef MLLM_CPUSPARSELINEAR_H
#define MLLM_CPUSPARSELINEAR_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUSparseLinear final : public Op {
public:
    CPUSparseLinear(Backend *bn, string opName, int in_dim, int out_dim, int threadCount);
    ~CPUSparseLinear() override = default;
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode load(AbstructLoader &loader) override;
    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int in_dim_;
    int out_dim_;
    int thread_count = 4;
    Tensor weight_; // weight of shape [in_dim_, out_dim_]. dst = x * weight (we use a different way to compute mat_mul)
};

class CPUSparseLinearCreator : public CPUBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        int in_dim_ = (int)op_param["in_dim_"];
        int out_dim_ = (int)op_param["out_dim_"];
        return new CPUSparseLinear(bn, name, in_dim_, out_dim_, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUSPARSELINEAR_H
