#ifndef MLLM_CPULINEARSPARSE_H
#define MLLM_CPULINEARSPARSE_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class Tensor;
class CPULinearSparse final : public Op {
public:
    CPULinearSparse(Backend *bn, string opName, int in_features, int out_features, bool bias, int threadCount);
    virtual ~CPULinearSparse() = default;
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
    AbstructLoader *loader_;

    ErrorCode dynamicLoad(AbstructLoader *loader, std::set<int> inputs);
};

class CPULinearSparseCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int in_features = op_param["in_features"];
        int out_features = op_param["out_features"];
        int bias = op_param["bias"];
        return new CPULinearSparse(bn, name, in_features, out_features, (bool)bias, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPULINEARSPARCE_H