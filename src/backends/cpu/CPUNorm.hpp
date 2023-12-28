
#ifndef MLLM_CPUNORM_H
#define MLLM_CPUNORM_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPUNorm final : public Op {
public:
    CPUNorm(Backend *bn, string opName, int L_n, bool multiThread);
    virtual ~CPUNorm() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    bool support_multi_thread_ = false;
    int L_n_ = 2;
};

class CPUNormCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        int L_n = op_param["L_n"];
        return new CPUNorm(bn, name, L_n, false);
    }
};

} // namespace mllm

#endif // MLLM_CPUNORM_H
