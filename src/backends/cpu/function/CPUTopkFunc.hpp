//
// Created by Rongjie Yi on 24-12-16.
//

#ifndef CPUTOPKFUNC_HPP
#define CPUTOPKFUNC_HPP
#include "CPUBackend.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include <cassert>
#include <queue>

namespace mllm {
class Tensor;

class CPUtopkFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        assert(args.size() == 2);
        int k = (int)args[0];
        Chl dim = (Chl)args[1];
        if (dim == DIMENSION) {
            outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), k);
            outputs[1]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), k);
        }
        outputs[0]->setDtype(inputs[0]->dtype()); // topk_values
        outputs[0]->alloc();
        outputs[1]->setDtype(inputs[0]->dtype()); // topk_indices
        outputs[1]->alloc();
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        int k = (int)args[0];
        Chl dim = (Chl)args[1];
        if (dim == DIMENSION) {
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
            for (int n = 0; n < inputs[0]->batch(); n++) {
                for (int h = 0; h < inputs[0]->head(); h++) {
                    for (int s = 0; s < inputs[0]->sequence(); s++) {
                        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<>> topk_value_indices;
                        for (int d = 0; d < inputs[0]->dimension(); ++d) {
                            float value = inputs[0]->dataAt<float>(n, h, s, d);
                            topk_value_indices.push({value, d});
                            if (topk_value_indices.size() > k) {
                                topk_value_indices.pop();
                            }
                        }
                        for (int d = k - 1; d >= 0; --d) {
                            auto top = topk_value_indices.top();
                            topk_value_indices.pop();
                            outputs[0]->setDataAt<float>(n, h, s, d, top.first);
                            outputs[1]->setDataAt<float>(n, h, s, d, top.second);
                        }
                    }
                }
            }
        }
    }
};
} // namespace mllm
#endif // CPUTOPKFUNC_HPP