//
// Created by 咸的鱼 on 2023/11/26.
//

#ifndef MLLM_CPURELU_HPP
#define MLLM_CPURELU_HPP

#include "Op.hpp"
#include "CPUBackend.hpp"
namespace mllm {
class CPUReLU final : public Op {
public:
    CPUReLU(Backend *bn, string opName, bool multiThread);
    virtual ~CPUReLU() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;


private:
    bool support_multi_thread_ = false;
};

class CPUReLUCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new CPUReLU(bn, name, false);
    }
};
} // namespace mllm

#endif // MLLM_CPURELU_HPP
