#ifndef MLLM_CPURMSNORM_H
#define MLLM_CPURMSNORM_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPURMSNorm final : public Op {
public:
    CPURMSNorm(Backend *bn, string opName, int normSize, float epsilon = 1e-6, bool add_unit_offset_ = false, int threadCount = 4);
    virtual ~CPURMSNorm() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    Tensor &weight() {
        return weight_;
    }

private:
    int thread_count = 4;
    float epsilon_;
    int axis_ = 1;
    Tensor weight_;
    int normSize_;
    bool add_unit_offset_;
    // Tensor bias_;
};

class CPURMSNormCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int normSize = (int)op_param["norm_size"];
        float epsilon = (float)op_param["epsilon"];
        bool add_unit_offset_ = (op_param.find("add_unit_offset") == op_param.end()) ? false : op_param["add_unit_offset"];
        return new CPURMSNorm(bn, name, normSize, epsilon, add_unit_offset_, threadCount);
    }
};
} // namespace mllm

#endif // MLLM_CPURMSNORM_H