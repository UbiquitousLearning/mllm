
#ifndef MLLM_CPUSHAPE_H
#define MLLM_CPUSHAPE_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUShape final : public Op {
public:
    CPUShape(Backend *bn, string opName, Chl axis, int threadCount);
    virtual ~CPUShape() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    Chl axis_;
    int thread_count = 4;
};

class CPUShapeCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        Chl axis = (Chl)op_param["axis"];
        return new CPUShape(bn, name, axis, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUSHAPE_H
