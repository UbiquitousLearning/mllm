//
// Created by 咸的鱼 on 2023/11/26.
//

#ifndef MLLM_CPURELU2_HPP
#define MLLM_CPURELU2_HPP

#include "Op.hpp"
#include "CPUBackend.hpp"
namespace mllm {
class CPUReLU2 final : public Op {
public:
    CPUReLU2(Backend *bn, string opName, bool multiThread);
    virtual ~CPUReLU2() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;


private:
    bool support_multi_thread_ = false;
};

class CPUReLU2Creator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new CPUReLU2(bn, name, false);
    }
};
} // namespace mllm

#endif // MLLM_CPURELU2_HPP
