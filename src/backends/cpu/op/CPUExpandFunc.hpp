//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUEXPANDFUNC_HPP
#define CPUEXPANDFUNC_HPP

#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"
#include <cstring>
#include <iostream>

namespace mllm {
class Tensor;

class CPUexpandFunction : public Op {
private:
    int thread_count = 4;
    int b_, h_, s_, d_;

public:
    CPUexpandFunction(Backend *bn, string name, int threadCount, int b, int h, int s, int d) :
        Op(bn, name), thread_count(threadCount), b_(b), h_(h), s_(s), d_(d) {
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        // The original assert seems to imply only one dimension can be expanded at a time.
        // Let's ensure a similar check but allow -1 for non-expanded dims.
        // Example: b=5, h=-1, s=-1, d=-1. (5 * -1 * -1 * -1) = -5 < 0. This logic is preserved.
        assert(b_ * h_ * s_ * d_ < 0);

        int dim_b = inputs[0]->batch();
        int dim_h = inputs[0]->head();
        int dim_s = inputs[0]->sequence();
        int dim_d = inputs[0]->dimension();

        if (b_ != -1) {
            assert(dim_b == 1);
            dim_b = b_;
        } else if (s_ != -1) {
            assert(dim_s == 1);
            dim_s = s_;
        } else if (h_ != -1) {
            assert(dim_h == 1);
            dim_h = h_;
        } else if (d_ != -1) {
            assert(dim_d == 1);
            dim_d = d_;
        }

        outputs[0]->reshape(dim_b, dim_h, dim_s, dim_d);
        outputs[0]->setDtype(inputs[0]->dtype());
        return ErrorCode::MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        int dim_b = inputs[0]->batch();
        int dim_s = inputs[0]->sequence();
        int dim_h = inputs[0]->head();
        int dim_d = inputs[0]->dimension();

        if (b_ != -1) {
            std::cerr << "expand for BATCH not support" << std::endl;
        } else if (s_ != -1) {
#pragma omp parallel for collapse(2) num_threads(thread_count)
            for (int b = 0; b < dim_b; ++b) {
                for (int s = 0; s < s_; ++s) {
                    memcpy(outputs[0]->ptrAt<char>(b, 0, s, 0),
                           inputs[0]->ptrAt<char>(b, 0, 0, 0),
                           dim_d * dim_h * inputs[0]->dtypeSize());
                }
            }
        } else if (h_ != -1) {
#pragma omp parallel for collapse(3) num_threads(thread_count)
            for (int b = 0; b < dim_b; ++b) {
                for (int s = 0; s < dim_s; ++s) {
                    for (int h = 0; h < h_; ++h) {
                        memcpy(outputs[0]->ptrAt<char>(b, h, s, 0),
                               inputs[0]->ptrAt<char>(b, 0, s, 0), // Assumes input head is 1
                               dim_d * inputs[0]->dtypeSize());
                    }
                }
            }
        } else if (d_ != -1) {
            for (int b = 0; b < dim_b; ++b) {
                for (int s = 0; s < dim_s; ++s) {
                    for (int h = 0; h < dim_h; ++h) {
                        float data = inputs[0]->dataAt<float>(b, h, s, 0);
                        std::fill_n(outputs[0]->ptrAt<float>(b, h, s, 0), outputs[0]->dimension(), data);
                    }
                }
            }
        }
        return ErrorCode::MLLM_NO_ERROR;
    }
};

class CPUexpandFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // Assumes OpParam contains keys "b", "h", "s", "d"
        int b = static_cast<int>(op_param.at("b"));
        int h = static_cast<int>(op_param.at("h"));
        int s = static_cast<int>(op_param.at("s"));
        int d = static_cast<int>(op_param.at("d"));
        return new CPUexpandFunction(bn, name, threadCount, b, h, s, d);
    }
};

} // namespace mllm
#endif // CPUEXPANDFUNC_HPP