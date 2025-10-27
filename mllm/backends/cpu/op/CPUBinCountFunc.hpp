//
// Created by Rongjie Yi on 24-12-16.
//

#ifndef CPUBINCOUNTKFUNC_HPP
#define CPUBINCOUNTKFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"
#include <algorithm>
#include <cassert>
#include "CPUBackend.hpp"

namespace mllm {
class Tensor;

class CPUbincountFunction : public Op {
private:
    int thread_count = 4;

    void bincount(float *input, int size, float *out, int max_val) {
        // 初始化输出数组
#pragma omp parallel for collapse(1) num_threads(CPUBackend::cpu_threads)
        for (int i = 0; i <= max_val; ++i) {
            out[i] = 0;
        }

        // 计算每个值的出现次数
        // #pragma omp parallel for collapse(1) num_threads(CPUBackend::cpu_threads)
        for (int i = 0; i < size; ++i) {
            int index = static_cast<int>(input[i]);
            if (index >= 0 && index <= max_val) {
                out[index] += 1;
            }
        }
    }

public:
    CPUbincountFunction(Backend *bn, string name, int threadCount) :
        thread_count(threadCount), Op(bn, name) {
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        assert(inputs[0]->batch() == 1);
        assert(inputs[0]->sequence() == 1);
        assert(inputs[0]->head() == 1);
        // For dynamic-shape ops, reshape sets what's known. Final shape is set in execute.
        outputs[0]->reshape(1, 1, 1, 0);
        outputs[0]->setDtype(MLLM_TYPE_F32);
        return ErrorCode::MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        int size = inputs[0]->dimension();
        int max_val = 0;
        for (int i = 0; i < size; ++i) {
            int val = static_cast<int>(inputs[0]->dataAt<float>(0, 0, 0, i));
            if (val > max_val) {
                max_val = val;
            }
        }
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), max_val + 1);
        outputs[0]->setDtype(inputs[0]->dtype()); // bincountk_values
        outputs[0]->alloc();
        float *data = inputs[0]->hostPtr<float>();
        float *out = outputs[0]->hostPtr<float>();
        if (max_val > 0) {
            bincount(data, size, out, max_val);
        }
        return ErrorCode::MLLM_NO_ERROR;
    }
};

class CPUbincountFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        return new CPUbincountFunction(bn, name, threadCount);
    }
};

} // namespace mllm
#endif // CPUBINCOUNTKFUNC_HPP