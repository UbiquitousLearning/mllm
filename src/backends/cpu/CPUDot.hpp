//
// Created by 30500 on 2023/10/12 0012.
//

#ifndef MLLM_CPUDOT_HPP
#define MLLM_CPUDOT_HPP

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {
class CPUDot final : public Op {
public:
    CPUDot(Backend *bn, string opName, bool multiThread);
    virtual ~CPUDot() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    
    virtual ErrorCode load(ParamLoader &loader) override;

private:
    bool support_multi_thread_ = false;
};

class CPUDotCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new CPUDot(bn, name, false);
    }
};
} // namespace mllm

#endif // MLLM_CPUDOT_HPP
