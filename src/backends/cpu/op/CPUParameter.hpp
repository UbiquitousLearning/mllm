
#ifndef MLLM_CPUPARAMETER_H
#define MLLM_CPUPARAMETER_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUParameter final : public Op {
public:
    CPUParameter(Backend *bn, string opName, int batch, int head, int seq, int dim, int threadCount);
    virtual ~CPUParameter() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    Tensor &weight() {
        return weight_;
    }

private:
    int thread_count = 4;
    Tensor weight_;
    int batch_;
    int head_;
    int seq_;
    int dim_;
};

class CPUParameterCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int batch = (int)op_param["batch"];
        int head = (int)op_param["head"];
        int seq = (int)op_param["seq"];
        int dim = (int)op_param["dim"];
        return new CPUParameter(bn, name, batch, head, seq, dim, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUPARAMETER_H
