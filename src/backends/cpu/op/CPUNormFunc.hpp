//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUNORMFUNC_HPP
#define CPUNORMFUNC_HPP

#include "CPUBackend.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include <cmath> // For std::sqrt and std::abs
#include <memory>

namespace mllm {
class Tensor;

class CPUnormFunction : public Op {
private:
    int thread_count = 4;
    int L_n_;

public:
    CPUnormFunction(Backend *bn, string name, int threadCount, int L_n)
        : Op(bn, name), thread_count(threadCount), L_n_(L_n) {}

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        // 遵从原始 reshape 逻辑，在这里 alloc
        // outputs[0]->alloc();
        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        // Parallelize the outer loops for better efficiency
        #pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int n = 0; n < inputs[0]->batch(); n++) {
            for (int h = 0; h < inputs[0]->head(); h++) {
                for (int s = 0; s < inputs[0]->sequence(); s++) {
                    if (L_n_ == 2) { // L2 Norm
                        float sum_of_squares = 0.0f;
                        for (int d = 0; d < inputs[0]->dimension(); ++d) {
                            float val = inputs[0]->dataAt<float>(n, h, s, d);
                            sum_of_squares += val * val;
                        }
                        float l2_norm = std::sqrt(sum_of_squares);
                        
                        // Broadcast the norm value across the dimension
                        for (int d = 0; d < inputs[0]->dimension(); d++) {
                            outputs[0]->setDataAt<float>(n, h, s, d, l2_norm);
                        }
                    } else { // L1 Norm (or other)
                        float sum_of_abs_values = 0.0f;
                        for (int d = 0; d < inputs[0]->dimension(); ++d) {
                            sum_of_abs_values += std::abs(inputs[0]->dataAt<float>(n, h, s, d));
                        }
                        
                        // Broadcast the norm value across the dimension
                        for (int d = 0; d < inputs[0]->dimension(); d++) {
                            outputs[0]->setDataAt<float>(n, h, s, d, sum_of_abs_values);
                        }
                    }
                }
            }
        }
        return MLLM_NO_ERROR;
    }
};

class CPUnormFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // Assumes OpParam contains the key "L_n"
        int L_n = static_cast<int>(op_param.at("L_n"));
        return new CPUnormFunction(bn, name, threadCount, L_n);
    }
};

} // namespace mllm
#endif // CPUNORMFUNC_HPP