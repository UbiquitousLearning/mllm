//
// Created by Xiang Li on 2023/11/26.
//

#ifndef MLLM_CPURELU2_HPP
#define MLLM_CPURELU2_HPP

#include "Op.hpp"
#include "../CPUBackend.hpp"
namespace mllm {
class CPUReLU2 final : public Op {
public:
    CPUReLU2(Backend *bn, string opName, int threadCount);
    virtual ~CPUReLU2() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
};

class CPUReLU2Creator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new CPUReLU2(bn, name, threadCount);
    }
};
} // namespace mllm

#endif // MLLM_CPURELU2_HPP
