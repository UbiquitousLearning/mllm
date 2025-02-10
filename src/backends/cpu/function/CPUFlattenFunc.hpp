//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUFLATTENFUNC_HPP
#define CPUFLATTENFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"
#include "Module.hpp"

namespace mllm {
class Tensor;

class CPUflattenFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        Chl axis_start = (Chl)args[0];
        Chl axis_end = (Chl)args[1];
        int dim_b = inputs[0]->batch();
        int dim_h = 0;
        int dim_s = 0;
        int dim_d = 0;
        if (inputs[0]->shape().size() == 4) {
            dim_h = inputs[0]->head();
            dim_s = inputs[0]->sequence();
            dim_d = inputs[0]->dimension();
            if (axis_start == BATCH & axis_end == SEQUENCE) {
                dim_b = 1;
                dim_s = inputs[0]->sequence() * inputs[0]->batch();
            } else if (axis_start == HEAD & axis_end == SEQUENCE) {
                dim_h = 1;
                dim_s = inputs[0]->sequence() * inputs[0]->head();
            } else if (axis_start == HEAD & axis_end == DIMENSION) {
                dim_h = 1;
                dim_d = inputs[0]->dimension() * inputs[0]->head();
            } else {
                std::cout << "ERROR:  flatten  " << axis_start << "&" << axis_end << std::endl;
            }
        } else if (inputs[0]->shape().size() == 5) {
            if (axis_start == CHANNLE & axis_end == HEIGHT) {
                dim_h = 1;
                dim_s = inputs[0]->channel() * inputs[0]->height() * inputs[0]->time();
                dim_d = inputs[0]->width();
            } else if (axis_start == HEIGHT & axis_end == CHANNLE) {
                dim_h = 1;
                dim_s = inputs[0]->channel() * inputs[0]->height() * inputs[0]->width();
                dim_d = inputs[0]->time();
            }
        }
        assert(dim_d + dim_s + dim_h > 0);
        outputs[0]->reshape(dim_b, dim_h, dim_s, dim_d);
        if ((axis_start == TIME & axis_end == WIDTH && inputs[0]->ctype() == BCTHW)
            || (axis_start == CHANNLE & axis_end == HEIGHT && inputs[0]->ctype() == BWCTH)
            || (axis_start == HEIGHT & axis_end == CHANNLE && inputs[0]->ctype() == BTHWC)
            || (axis_start == BATCH & axis_end == SEQUENCE && inputs[0]->ctype() != BCTHW)
            || (axis_start == HEAD & axis_end == SEQUENCE && inputs[0]->ctype() == BSHD)
            || (axis_start == HEAD & axis_end == SEQUENCE && inputs[0]->ctype() == BHDS)
            || (axis_start == HEAD & axis_end == SEQUENCE && inputs[0]->ctype() == BDHS)
            || (axis_start == HEAD & axis_end == DIMENSION && inputs[0]->ctype() == BSHD)
            || (axis_start == HEAD & axis_end == DIMENSION && inputs[0]->ctype() == BHDS)
            || (axis_start == HEAD & axis_end == SEQUENCE && inputs[0]->ctype() == BDSH)) {
            if (inputs[0]->masterTensor() == nullptr) {
                inputs[0]->free();
            }
            outputs[0]->setDtype(inputs[0]->dtype());
            outputs[0]->alloc();
            inputs[0]->shallowCopyFrom(outputs[0], false);
        } else if (Module::llm_model_ptr->op_transposed_flag) {
            if (inputs[0]->masterTensor() == nullptr) {
                inputs[0]->free();
            }
            outputs[0]->setDtype(inputs[0]->dtype());
            outputs[0]->alloc();
            inputs[0]->shallowCopyFrom(outputs[0], false);
            return;
        } else {
            std::cout << "[TODO]Tensor.Flatten not support!!!!" << std::endl;
        }
    }

    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
    }
};

} // namespace mllm
#endif // CPUFLATTENFUNC_HPP