#ifndef MLLM_CPURMSNORM_H
#define MLLM_CPURMSNORM_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPURMSNorm final : public Op {
public:
    CPURMSNorm(Backend *bn, string opName, bool multiThread, float epsilon = 1e-5);
    virtual ~CPURMSNorm() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(ParamLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    Tensor &weight() {
        return weight_;
    }

private:
    bool support_multi_thread_ = false;
    float epsilon_;
    int axis_ = 1;
    Tensor weight_;
    int normSize_;
    // Tensor bias_;
};

class CPURMSNormCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new CPURMSNorm(bn, name, false);
    }
};
} // namespace mllm

#endif // MLLM_CPURMSNORM_H