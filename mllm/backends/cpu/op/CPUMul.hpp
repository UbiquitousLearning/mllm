//
// Created by Rongjie Yi on 2023/10/12 0012.
//

#ifndef MLLM_CPUMUL_HPP
#define MLLM_CPUMUL_HPP

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {
class CPUMul final : public Op {
public:
    CPUMul(Backend *bn, string opName, int threadCount);
    virtual ~CPUMul() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
};

class CPUMulCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new CPUMul(bn, name, threadCount);
    }
};
} // namespace mllm

#endif // MLLM_CPUMUL_HPP
