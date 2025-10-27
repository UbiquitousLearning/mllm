/**
 * @file CPUSlidingWindowMask.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief sliding window mask for SWA(sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".)
 * @version 0.1
 * @date 2024-04-27
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef MLLM_CPUSLIDINGWINDOWMASK_H
#define MLLM_CPUSLIDINGWINDOWMASK_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUSlidingWindowMask final : public Op {
public:
    CPUSlidingWindowMask(Backend *bn, string opName, int windowSize, int threadCount);
    ~CPUSlidingWindowMask() override = default;
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode load(AbstructLoader &loader) override;
    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
    int winodw_size;
};

class CPUSlidingWindowMaskCreator : public CPUBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        int windowSize = (int)op_param["window_size"];
        return new CPUSlidingWindowMask(bn, name, windowSize, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUSLIDINGWINDOWMASK_H
