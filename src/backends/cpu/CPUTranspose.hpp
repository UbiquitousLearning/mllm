
#ifndef MLLM_CPUTRANSPOSE_H
#define MLLM_CPUTRANSPOSE_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPUTranspose final : public Op {
public:
    CPUTranspose(Backend *bn, string opName, bool multiThread);
    virtual ~CPUTranspose() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    bool support_multi_thread_ = false;
};

class CPUTransposeCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new CPUTranspose(bn, name, false);
    }
};

} // namespace mllm

#endif // MLLM_CPUTRANSPOSE_H
