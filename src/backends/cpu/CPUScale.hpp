#ifndef MLLM_CPUSCALE_H
#define MLLM_CPUSCALE_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPUScale : public Op {
public:
    CPUScale(Backend *bn, bool multiThread);
    virtual ~CPUScale() = default;
    virtual ErrorCode Reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
    virtual ErrorCode Setup(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
    virtual ErrorCode Execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;

    virtual ErrorCode Load(ParamLoader &loader) override;

private:
    bool support_multi_thread_ = false;
};

class CPUScaleCreator : public CPUBackend::Creator {
public:
    virtual Op *Create(OpParam op_param, Backend *bn) const {
        return new CPUScale(bn, false);
    }
};

} // namespace mllm

#endif // MLLM_CPUSCALE_H