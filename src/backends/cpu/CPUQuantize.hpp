//
// Created by Xiang Li on 2023/11/26.
//

#ifndef MLLM_CPUQUANTIZE_HPP
#define MLLM_CPUQUANTIZE_HPP

#include "Op.hpp"
#include "CPUBackend.hpp"
namespace mllm {
class CPUQuantize final : public Op {
public:
    CPUQuantize(Backend *bn, string opName, int threadCount);
    virtual ~CPUQuantize() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;


private:
    int thread_count = 4;
    Tensor scale_;
};

class CPUQuantizeCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new CPUQuantize(bn, name, threadCount);
    }
};
} // namespace mllm

#endif // MLLM_CPUQUANTIZE_HPP
