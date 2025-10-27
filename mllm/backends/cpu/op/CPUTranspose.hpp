
#ifndef MLLM_CPUTRANSPOSE_H
#define MLLM_CPUTRANSPOSE_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUTranspose final : public Op {
public:
    CPUTranspose(Backend *bn, string opName, int axis0, int axis1, int threadCount);
    virtual ~CPUTranspose() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    Chl axis0_;
    Chl axis1_;
    int thread_count = 4;
};

class CPUTransposeCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int axis0 = (int)op_param["axis0"];
        int axis1 = (int)op_param["axis1"];
        return new CPUTranspose(bn, name, axis0, axis1, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUTRANSPOSE_H
