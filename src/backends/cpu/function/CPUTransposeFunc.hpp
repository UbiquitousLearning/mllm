//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUTRANSPOSEFUNC_HPP
#define CPUTRANSPOSEFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"
#include "Module.hpp"

namespace mllm {
class Tensor;

class CPUtransposeFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        vector<std::pair<Chl, Chl>> axiss;
        for (int i = 0; i < args.size(); i += 2) {
            axiss.push_back({(Chl)args[i], (Chl)args[i + 1]});
        }
        // if (outputs[0]->count() <= 0 || outputs[0]->shape() != inputs[0]->shape())
        // {
        outputs[0]->transCopyShape(inputs[0]->shape());
        if (!Module::llm_model_ptr->op_transposed_flag) {
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
        // if (inputs[0]->masterTensor() != nullptr) {
        if (inputs[0]->masterTensor() != nullptr && (inputs[0]->masterTensor()->name().find("Cache") != std::string::npos || inputs[0]->masterTensor()->name().find("weight") != std::string::npos)) {
            if (outputs[0]->masterTensor() == nullptr) {
                outputs[0]->setDtype(inputs[0]->dtype());
                outputs[0]->shallowCopyFrom(inputs[0], false);
            }
        } else {
            if (inputs[0]->masterTensor() == nullptr) {
                inputs[0]->free();
            }
            outputs[0]->setDtype(inputs[0]->dtype());
            outputs[0]->alloc();
            // inputs[0]->undiffusion() = true;
            inputs[0]->setUndiffusion(true);
            inputs[0]->shallowCopyFrom(outputs[0], false);
            outputs[0]->transFrom() = axiss;
        }
        // }
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
    }
};
} // namespace mllm
#endif // CPUTRANSPOSEFUNC_HPP