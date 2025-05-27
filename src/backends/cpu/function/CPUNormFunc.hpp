//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUNORMFUNC_HPP
#define CPUNORMFUNC_HPP
#include "CPUBackend.hpp"
#include "Tensor.hpp"
#include "Types.hpp"

namespace mllm {
class Tensor;

class CPUnormFunction : public TensorFunction {
public:
    void reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override {
        int L_n = (int)args[0];
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
    }
    void execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override {
        int L_n = (int)args[0];
        for (int h = 0; h < inputs[0]->head(); h++) {
            for (int n = 0; n < inputs[0]->batch(); n++) {
                for (int s = 0; s < inputs[0]->sequence(); s++) {
                    if (L_n == 2) {
                        float sum_of_squares = 0.0f;
                        for (int d = 0; d < inputs[0]->dimension(); ++d) {
                            sum_of_squares += inputs[0]->dataAt<float>(n, h, s, d) * inputs[0]->dataAt<float>(n, h, s, d);
                        }
                        float l2_norm = std::sqrt(sum_of_squares);
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
                        for (int d = 0; d < inputs[0]->dimension(); d++) {
                            outputs[0]->setDataAt<float>(n, h, s, d, l2_norm);
                        }
                    } else {
                        float sum_of_abs_values = 0.0f;
                        for (int d = 0; d < inputs[0]->dimension(); ++d) {
                            sum_of_abs_values += std::abs(inputs[0]->dataAt<float>(n, h, s, d));
                        }
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
                        for (int d = 0; d < inputs[0]->dimension(); d++) {
                            outputs[0]->setDataAt<float>(n, h, s, d, sum_of_abs_values);
                        }
                    }
                }
            }
        }
    }
};
} // namespace mllm
#endif // CPUNORMFUNC_HPP