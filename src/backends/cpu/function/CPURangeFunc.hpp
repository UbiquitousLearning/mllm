//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPURANGEFUNC_HPP
#define CPURANGEFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"

namespace mllm {
class Tensor;

class CPURangeFunction : public TensorFunction {
public:
    void reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override {
        int start = (int)args[0];
        int end = (int)args[1];
        outputs[0]->reshape(1, 1, end - start, 1);
        outputs[0]->setDtype(MLLM_TYPE_F32);
        outputs[0]->alloc();
    }
    void execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override {
        int start = (int)args[0];
        int end = (int)args[1];
#pragma omp parallel for collapse(1) num_threads(CPUBackend::cpu_threads)
        for (int i = 0; i < end - start; ++i) {
            outputs[0]->setDataAt<float>(0, 0, i + start, 0, (float)i);
        }
    }
};

} // namespace mllm
#endif // CPURANGEFUNC_HPP