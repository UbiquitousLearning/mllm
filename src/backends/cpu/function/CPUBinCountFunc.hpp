//
// Created by Rongjie Yi on 24-12-16.
//

#ifndef CPUBINCOUNTKFUNC_HPP
#define CPUBINCOUNTKFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"
#include <cassert>
#include "CPUBackend.hpp"

namespace mllm {
class Tensor;

class CPUbincountFunction : public TensorFunction {
    void bincount(float *input, int size, float *out, int max_val) {
        // 找到输入数组中的最大值
        // int max_val = 0;
        // for (int i = 0; i < size; ++i) {
        //     int val = static_cast<int>(input[i]);
        //     if (val > max_val) {
        //         max_val = val;
        //     }
        // }

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
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        assert(args.empty());
        assert(inputs[0]->batch() == 1);
        assert(inputs[0]->sequence() == 1);
        assert(inputs[0]->head() == 1);
        outputs[0]->reshape(1, 1, 1, 0);
        // outputs[0]->setDtype(inputs[0]->dtype()); // bincountk_values
        outputs[0]->alloc();
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
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
    }
};
} // namespace mllm
#endif // CPUBINCOUNTKFUNC_HPP