
#ifndef MLLM_CPUPREDICTOR_H
#define MLLM_CPUPREDICTOR_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUPredictor final : public Op {
public:
    CPUPredictor(Backend *bn, string name, int in_dim, int out_dim, int threadCount);
    ~CPUPredictor() override = default;
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode load(AbstructLoader &loader) override;
    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    /*
     * two linear, in_dim_ => r_ => out_dim_
     * r can be computed using the weight_size
     * */
    int in_dim_;
    int out_dim_;
    int r_;
    int thread_count = 4;
    Tensor up_;
    Tensor down_;
    Tensor hidden_; // store hidden activation
};

class CPUPredictorCreator : public CPUBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        return new CPUPredictor(bn, std::move(name),
                                (int)op_param["in_dim"],
                                (int)op_param["out_dim"],
                                threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUPREDICTOR_H
