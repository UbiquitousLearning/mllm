//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPURANGEFUNC_HPP
#define CPURANGEFUNC_HPP

#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"
#include <memory>

namespace mllm {
class Tensor;

class CPURangeFunction : public Op {
private:
    int thread_count = 4;
    int start_;
    int end_;

public:
    CPURangeFunction(Backend *bn, string name, int threadCount, int start, int end)
        : Op(bn, name), thread_count(threadCount), start_(start), end_(end) {}

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        outputs[0]->reshape(1, 1, end_ - start_, 1);
        outputs[0]->setDtype(MLLM_TYPE_F32);
        // 遵从原始 reshape 逻辑，在这里 alloc
        // outputs[0]->alloc();
        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        int length = end_ - start_;
        #pragma omp parallel for num_threads(thread_count)
        for (int i = 0; i < length; ++i) {
            // Bug fix: Index should be 'i', value should be 'start_ + i'.
            // Original code had `setDataAt(..., i + start_, ..., (float)i)`, which was incorrect.
            outputs[0]->setDataAt<float>(0, 0, i, 0, (float)(start_ + i));
        }
        return MLLM_NO_ERROR;
    }
};

class CPURangeFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // Assumes OpParam contains keys "start" and "end"
        int start = static_cast<int>(op_param.at("start"));
        int end = static_cast<int>(op_param.at("end"));
        return new CPURangeFunction(bn, name, threadCount, start, end);
    }
};

} // namespace mllm
#endif // CPURANGEFUNC_HPP