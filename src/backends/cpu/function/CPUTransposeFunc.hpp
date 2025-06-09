//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUTRANSPOSEFUNC_HPP
#define CPUTRANSPOSEFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"
#include "Module.hpp"
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
    }
};
} // namespace mllm
#endif // CPUTRANSPOSEFUNC_HPP