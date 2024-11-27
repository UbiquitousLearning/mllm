//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUEXPANDFUNC_HPP
#define CPUEXPANDFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"

namespace mllm {
class Tensor;

class CPUexpandFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        int b = (int)args[0];
        int h = (int)args[1];
        int s = (int)args[2];
        int d = (int)args[3];
        assert(b * h * d * s < 0);
        int dim_b = inputs[0]->batch();
        int dim_h = inputs[0]->head();
        int dim_s = inputs[0]->sequence();
        int dim_d = inputs[0]->dimension();
        if (b != -1) {
            assert(dim_b == 1);
            dim_b = b;
        } else if (s != -1) {
            assert(dim_s == 1);
            dim_s = s;
        } else if (h != -1) {
            assert(dim_h == 1);
            dim_h = h;
        } else if (d != -1) {
            assert(dim_d == 1);
            dim_d = d;
        }
        outputs[0]->reshape(dim_b, dim_h, dim_s, dim_d);
        outputs[0]->alloc();
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        int b = (int)args[0];
        int h = (int)args[1];
        int s = (int)args[2];
        int d = (int)args[3];
        int dim_b = inputs[0]->batch();
        int dim_s = inputs[0]->sequence();
        int dim_h = inputs[0]->head();
        int dim_d = inputs[0]->dimension();
        if (b != -1) {
            std::cerr << "expand tp support" << std::endl;
        } else if (s != -1) {
#pragma omp parallel for collapse(2) num_threads(CPUBackend::cpu_threads)
            for (int b_ = 0; b_ < dim_b; ++b_) {
                for (int s_ = 0; s_ < s; ++s_) {
                    memcpy(outputs[0]->ptrAt<float>(b_, 0, s_, 0),
                           inputs[0]->ptrAt<float>(b_, 0, 0, 0),
                           dim_d * dim_h * sizeof(float));
                }
            }
        } else if (h != -1) {
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
            for (int b_ = 0; b_ < dim_b; ++b_) {
                for (int s_ = 0; s_ < dim_s; ++s_) {
                    for (int h_ = 0; h_ < h; ++h_) {
                        memcpy(outputs[0]->ptrAt<float>(b_, h_, s_, 0),
                               inputs[0]->ptrAt<float>(b_, h_, 0, 0),
                               dim_d * sizeof(float));
                    }
                }
            }
        } else if (d != -1) {
            std::cerr << "expand tp support" << std::endl;
        }
    }
};

} // namespace mllm
#endif // CPUEXPANDFUNC_HPP