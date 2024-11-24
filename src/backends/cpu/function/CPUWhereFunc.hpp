//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUWHEREFUNC_HPP
#define CPUWHEREFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"

namespace mllm {
class Tensor;

class CPUwhereFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        float value = args[0];
        Chl axis = (Chl)args[1];
        vector<float> b_vec = {};
        vector<float> s_vec = {};
        vector<float> h_vec = {};
        vector<float> d_vec = {};
        if (inputs[0]->count() % CPUBackend::cpu_threads == 0) {
#pragma omp parallel for collapse(4) num_threads(CPUBackend::cpu_threads)
            for (int b = 0; b < inputs[0]->batch(); b++) {
                for (auto s = 0; s < inputs[0]->sequence(); s++) {
                    for (auto h = 0; h < inputs[0]->head(); h++) {
                        for (auto d = 0; d < inputs[0]->dimension(); d++) {
                            if (inputs[0]->dataAt<float>(b, h, s, d) == value) {
                                b_vec.push_back(b);
                                s_vec.push_back(s);
                                h_vec.push_back(h);
                                d_vec.push_back(d);
                            }
                        }
                    }
                }
            }
        } else {
            for (int b = 0; b < inputs[0]->batch(); b++) {
                for (auto s = 0; s < inputs[0]->sequence(); s++) {
                    for (auto h = 0; h < inputs[0]->head(); h++) {
                        for (auto d = 0; d < inputs[0]->dimension(); d++) {
                            if (inputs[0]->dataAt<float>(b, h, s, d) == value) {
                                b_vec.push_back(b);
                                s_vec.push_back(s);
                                h_vec.push_back(h);
                                d_vec.push_back(d);
                            }
                        }
                    }
                }
            }
        }
        int num = b_vec.size();
        if ((int)axis == -1) {
            outputs[0]->reshape(1, 1, 4, num);
            outputs[0]->setDtype(inputs[0]->dtype());
            outputs[0]->alloc();
            for (int i = 0; i < 4; ++i) {
                auto dest_ptr = outputs[0]->hostPtr<float>() + outputs[0]->offset(0, 0, i, 0);
                switch (i) {
                case 0:
                    memcpy(dest_ptr, b_vec.data(), num * sizeof(float));
                    break;
                case 1:
                    memcpy(dest_ptr, h_vec.data(), num * sizeof(float));
                    break;
                case 2:
                    memcpy(dest_ptr, s_vec.data(), num * sizeof(float));
                    break;
                case 3:
                    memcpy(dest_ptr, d_vec.data(), num * sizeof(float));
                    break;
                default:
                    break;
                }
            }
        } else {
            outputs[0]->reshape(1, 1, 1, num);
            outputs[0]->setDtype(inputs[0]->dtype());
            outputs[0]->alloc();
            auto dest_ptr = outputs[0]->hostPtr<float>();
            switch (axis) {
            case BATCH:
                memcpy(dest_ptr, b_vec.data(), num * sizeof(float));
                break;
            case HEAD:
                memcpy(dest_ptr, h_vec.data(), num * sizeof(float));
                break;
            case SEQUENCE:
                memcpy(dest_ptr, s_vec.data(), num * sizeof(float));
                break;
            case DIMENSION:
                memcpy(dest_ptr, d_vec.data(), num * sizeof(float));
                break;
            default:
                break;
            }
        }
    }
};

} // namespace mllm
#endif // CPUWHEREFUNC_HPP