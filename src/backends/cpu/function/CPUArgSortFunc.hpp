//
// Created by Rongjie Yi on 24-12-16.
//

#ifndef CPUARGSORTKFUNC_HPP
#define CPUARGSORTKFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"
#include <cassert>

namespace mllm {
class Tensor;

class CPUargsortFunction : public TensorFunction {
    // 自定义比较函数，用于对索引进行排序
    bool compareIndices(const std::pair<int, float> &a, const std::pair<int, float> &b) {
        return a.second < b.second;
    }
    void argsort(float *input, int size, float *out_indices) {
        std::vector<std::pair<int, float>> indexedInput(size);
        for (int i = 0; i < size; ++i) {
            indexedInput[i] = std::make_pair(i, input[i]);
        }
        std::sort(indexedInput.begin(), indexedInput.end(), [this](const std::pair<int, float> &a, const std::pair<int, float> &b) {
            return compareIndices(a, b);
        });
        for (int i = 0; i < size; ++i) {
            out_indices[i] = static_cast<float>(indexedInput[i].first);
        }
    }

public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        assert(args.empty());
        assert(inputs[0]->sequence() == 1);
        assert(inputs[0]->head() == 1);
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype()); // argsortk_values
        outputs[0]->alloc();
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        int size = inputs[0]->dimension();
        for (int b = 0; b < inputs[0]->batch(); b++) {
            float *data = inputs[0]->ptrAt<float>(b, 0, 0, 0);
            float *out = outputs[0]->ptrAt<float>(b, 0, 0, 0);
            argsort(data, size, out);
        }
    }
};
} // namespace mllm
#endif // CPUARGSORTKFUNC_HPP