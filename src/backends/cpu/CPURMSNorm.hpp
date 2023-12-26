#ifndef MLLM_CPURMSNORM_H
#define MLLM_CPURMSNORM_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPURMSNorm final : public Op {
public:
    CPURMSNorm(Backend *bn, string opName,int normSize, float epsilon = 1e-6,  bool multiThread= true);
    virtual ~CPURMSNorm() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
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
        int normSize = (int)op_param["norm_size"];
        return new CPURMSNorm(bn, name, normSize);
    }
};
} // namespace mllm

#endif // MLLM_CPURMSNORM_H