//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUTRANSPOSEFUNC_HPP
#define CPUTRANSPOSEFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"
#include "Module.hpp"
#include "compute/Quantize.hpp"
#include <cassert>
#include <ostream>

namespace mllm {
class Tensor;

class CPUtransposeFunction : public TensorFunction {
public:
    void setUp(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override {
        vector<std::pair<Chl, Chl>> axiss;
        for (int i = 0; i < args.size(); i += 2) {
            axiss.push_back({(Chl)args[i], (Chl)args[i + 1]});
        }
        // for BSHD attention start
        if(axiss.size() == 1&& axiss[0].first == HEAD && axiss[0].second == SEQUENCE) {
            if(inputs[0]->ctype() == BSHD){
                outputs[0]->chls() = {{BATCH, 0}, {HEAD, 1}, {SEQUENCE, 2}, {DIMENSION, 3}};
            }else{
                outputs[0]->chls() = {{BATCH, 0}, {SEQUENCE, 1}, {HEAD, 2}, {DIMENSION, 3}};
            }
            outputs[0]->changeCtype(4); 
            outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
            return;
        }
        // for BSHD attention swnd
        if (!outputs[0]->undiffusion()) {
            outputs[0]->transCopyShape(inputs[0]->shape());
            std::map<Chl, int> origin_chls = {{BATCH, 0}, {SEQUENCE, 1}, {HEAD, 2}, {DIMENSION, 3}, {CHANNLE, 1}, {TIME, 2}, {HEIGHT, 3}, {WIDTH, 4}};
            if (std::equal(outputs[0]->chls().begin(), outputs[0]->chls().end(), origin_chls.begin())) {
                outputs[0]->chls() = inputs[0]->chls();
                for (auto axis : axiss) {
                    auto axis0 = axis.first;
                    auto axis1 = axis.second;
                    auto ori_0_idx = outputs[0]->chls()[axis0];
                    auto ori_1_idx = outputs[0]->chls()[axis1];
                    outputs[0]->chls()[axis0] = ori_1_idx;
                    outputs[0]->chls()[axis1] = ori_0_idx;
                }
                outputs[0]->changeCtype(inputs[0]->shape().size());
                outputs[0]->undiffusion() = true;
            }
        }
        if (inputs[0]->masterTensor() != nullptr && (inputs[0]->masterTensor()->name().find("Cache") != std::string::npos || inputs[0]->masterTensor()->name().find("weight") != std::string::npos)) {
            if (outputs[0]->masterTensor() == nullptr) {
                outputs[0]->setDtype(inputs[0]->dtype());
                outputs[0]->shallowCopyFrom(inputs[0].get(), false);
            }
        } else {
            if (inputs[0]->masterTensor() == nullptr) {
                inputs[0]->free();
            }
            outputs[0]->setDtype(inputs[0]->dtype());
            outputs[0]->alloc();
            inputs[0]->setUndiffusion(true);
            inputs[0]->shallowCopyFrom(outputs[0].get(), false);
            outputs[0]->transFrom() = axiss;
        }
    }
    void reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override {
        vector<std::pair<Chl, Chl>> axiss;
        for (int i = 0; i < args.size(); i += 2) {
            axiss.push_back({(Chl)args[i], (Chl)args[i + 1]});
        }
        // for BSHD attention start
        if(axiss.size() == 1&& axiss[0].first == HEAD && axiss[0].second == SEQUENCE) {
            return;
        }
        // for BSHD attention send
        std::map<Chl, int> origin_chls = {{BATCH, 0}, {SEQUENCE, 1}, {HEAD, 2}, {DIMENSION, 3}, {CHANNLE, 1}, {TIME, 2}, {HEIGHT, 3}, {WIDTH, 4}};
        auto origin_s = inputs[0]->shape().size();
        outputs[0]->transCopyShape(inputs[0]->shape());
        if (inputs[0]->masterTensor() == nullptr
            || std::equal(outputs[0]->chls().begin(), outputs[0]->chls().end(), origin_chls.begin())) {
            outputs[0]->chls() = inputs[0]->chls();
            for (auto axis : axiss) {
                auto axis0 = axis.first;
                auto axis1 = axis.second;
                auto ori_0_idx = outputs[0]->chls()[axis0];
                auto ori_1_idx = outputs[0]->chls()[axis1];
                outputs[0]->chls()[axis0] = ori_1_idx;
                outputs[0]->chls()[axis1] = ori_0_idx;
            }
            outputs[0]->changeCtype(origin_s);
            outputs[0]->undiffusion() = true;
        }
        if (inputs[0]->masterTensor() != nullptr
            && (inputs[0]->masterTensor()->name().find("Cache") != std::string::npos || inputs[0]->masterTensor()->name().find("weight") != std::string::npos)) {
            // outputs[0]->shallowCopyFrom(inputs[0]->masterTensor(), false, inputs[0]->shapeOffset());
            if (outputs[0]->masterTensor() == nullptr) {
                outputs[0]->setDtype(inputs[0]->dtype());
                outputs[0]->shallowCopyFrom(inputs[0].get(), false);
            }
        }
    }
    void execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override {
                vector<std::pair<Chl, Chl>> axiss;
        // for BSHD attention start
        for (int i = 0; i < args.size(); i += 2) {
            axiss.push_back({(Chl)args[i], (Chl)args[i + 1]});
        }
        if(axiss.size() == 1&& axiss[0].first == HEAD && axiss[0].second == SEQUENCE) {
            if(inputs[0]->ctype() == BSHD){
                outputs[0]->chls() = {{BATCH, 0}, {HEAD, 1}, {SEQUENCE, 2}, {DIMENSION, 3}};
            }else{
                outputs[0]->chls() = {{BATCH, 0}, {SEQUENCE, 1}, {HEAD, 2}, {DIMENSION, 3}};
            }
            outputs[0]->changeCtype(4); 
            outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), 
                            inputs[0]->sequence(), inputs[0]->dimension());
            outputs[0]->alloc();
            //BSHD -> BSHD
            { //真转置
                assert(inputs[0]->batch() == 1);
                assert(outputs[0]->batch() == 1);
                assert(inputs[0]->head() == outputs[0]->head());
                assert(inputs[0]->sequence() == outputs[0]->sequence());
                assert(outputs[0]->ctype() == BHSD || outputs[0]->ctype() == BSHD);
                if(inputs[0]->dtype() == outputs[0]->dtype()){
                    #pragma omp parallel for
                    for (int h = 0; h < inputs[0]->head(); ++h) {
                        for (int s = 0; s < inputs[0]->sequence(); ++s) {
                            auto input_ptr = inputs[0]->ptrAt<float>(0, h, s, 0);
                            auto output_ptr = outputs[0]->ptrAt<float>(0, h, s, 0);
                            memcpy(output_ptr, input_ptr, inputs[0]->dimension() * sizeof(float));
                        }
                    }
                }else{
                    #pragma omp parallel for
                    for (int h = 0; h < inputs[0]->head(); ++h) {
                        for (int s = 0; s < inputs[0]->sequence(); ++s) {
                            auto input_ptr = inputs[0]->ptrAt<float>(0, h, s, 0);
                            auto output_ptr = outputs[0]->ptrAt<mllm_fp16_t>(0, h, s, 0);
                            for (int d = 0; d < inputs[0]->dimension(); ++d) {
                                output_ptr[d] = MLLM_FP32_TO_FP16(input_ptr[d]);
                            }
                        }
                    }
                }
            }
        }
        // for BSHD attention end
    }
};
} // namespace mllm
#endif // CPUTRANSPOSEFUNC_HPP