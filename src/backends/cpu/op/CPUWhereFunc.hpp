//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUWHEREFUNC_HPP
#define CPUWHEREFUNC_HPP

#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"
#include <vector>
#include <memory>

namespace mllm {
class Tensor;

class CPUwhereFunction : public Op {
private:
    int thread_count = 4;
    float value_;
    Chl axis_;

public:
    CPUwhereFunction(Backend *bn, string name, int threadCount, float value, Chl axis)
        : Op(bn, name), thread_count(threadCount), value_(value), axis_(axis) {}

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        // Shape is data-dependent and will be determined in execute.
        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        std::vector<float> b_vec;
        std::vector<float> s_vec;
        std::vector<float> h_vec;
        std::vector<float> d_vec;
        
        // NOTE: The original parallel implementation was thread-unsafe due to race conditions
        // on shared vectors. Using the sequential version for correctness.
        for (int b = 0; b < inputs[0]->batch(); b++) {
            for (auto s = 0; s < inputs[0]->sequence(); s++) {
                for (auto h = 0; h < inputs[0]->head(); h++) {
                    for (auto d = 0; d < inputs[0]->dimension(); d++) {
                        if (inputs[0]->dataAt<float>(b, h, s, d) == value_) {
                            b_vec.push_back(b);
                            s_vec.push_back(s);
                            h_vec.push_back(h);
                            d_vec.push_back(d);
                        }
                    }
                }
            }
        }
        
        int num = b_vec.size();
        if ((int)axis_ == -1) {
            outputs[0]->reshape(1, 1, 4, num);
            outputs[0]->setDtype(MLLM_TYPE_F32);
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
            outputs[0]->setDtype(MLLM_TYPE_F32);
            outputs[0]->alloc();
            auto dest_ptr = outputs[0]->hostPtr<float>();
            switch (axis_) {
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
        return MLLM_NO_ERROR;
    }
};

class CPUwhereFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        float value = op_param.at("value");
        Chl axis = (Chl)op_param.at("axis");
        return new CPUwhereFunction(bn, name, threadCount, value, axis);
    }
};

} // namespace mllm
#endif // CPUWHEREFUNC_HPP