//
// Created by 30500 on 2023/10/12 0012.
//

#ifndef MLLM_CPUMUL_HPP
#define MLLM_CPUMUL_HPP

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {
class CPUMul final : public Op {
public:
    CPUMul(Backend *bn, string opName, bool multiThread);
    virtual ~CPUMul() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    
    virtual ErrorCode load(ParamLoader &loader) override;

private:
    bool support_multi_thread_ = false;
};

class CPUMulCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new CPUMul(bn, name, false);
    }
};
} // namespace mllm

#endif // MLLM_CPUMUL_HPP
