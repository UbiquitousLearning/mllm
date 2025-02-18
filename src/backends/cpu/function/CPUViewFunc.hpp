//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUVIEWFUNC_HPP
#define CPUVIEWFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"

namespace mllm {
class Tensor;

class CPUviewFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        int b = (int)args[0];
        int h = (int)args[1];
        int s = (int)args[2];
        int d = (int)args[3];
        int dim_b = inputs[0]->batch();
        int dim_h = inputs[0]->head();
        int dim_s = inputs[0]->sequence();
        int dim_d = inputs[0]->dimension();
        if (b == -1 && h == 1 && s == 1 && d == -1) { //  sequence & head & dimension -> dimension
            dim_s = 1;
            dim_h = 1;
            dim_d = inputs[0]->sequence() * inputs[0]->head() * inputs[0]->dimension();
        } else if (b == -1 && h == -1 && s == 1 && d == 1) { //  sequence & head & dimension -> sequence
            dim_s = inputs[0]->sequence() * inputs[0]->head() * inputs[0]->dimension();
            dim_h = 1;
            dim_d = 1;
        } else if (b == 1 && h == 1 && s == -1 && d != -1 && inputs[0]->ctype() == BCTHW) { //  batch & head & sequence -> sequence
            dim_b = 1;
            dim_s = inputs[0]->channel() * inputs[0]->time() * inputs[0]->batch()* inputs[0]->height() *inputs[0]->width() /d;
            dim_h = 1;
            dim_d = d;
        } else if (b == 1 && h == 1 && s == -1 && d != -1) { //  batch & head & sequence -> sequence
            dim_b = 1;
            dim_s = inputs[0]->sequence() * inputs[0]->batch()* inputs[0]->head() *inputs[0]->dimension() /d;
            dim_h = 1;
            dim_d = d;
        } else if (b == -1 && h != -1 && s == -1 && d != -1) { // head & dimension
            if (h != ANYDIM && d != ANYDIM) {
                assert(inputs[0]->dimension() * inputs[0]->head() == h * d);
                dim_h = h;
                dim_d = d;
            } else if (h != ANYDIM) {
                dim_h = h;
                dim_d = inputs[0]->dimension() * inputs[0]->head() / h;
            } else if (d != ANYDIM) {
                dim_h = inputs[0]->dimension() * inputs[0]->head() / d;
                dim_d = d;
            } else {
                std::cout << "[TODO]Tensor.View not support!!!!" << std::endl;
            }
        } else if (b == -1 && h != -1 && s != -1 && d == -1) { // head & sequence
            if (h != ANYDIM && s != ANYDIM) {
                assert(inputs[0]->sequence() * inputs[0]->head() == h * s);
                dim_h = h;
                dim_s = s;
            } else if (h != ANYDIM) {
                dim_h = h;
                dim_s = inputs[0]->sequence() * inputs[0]->head() / h;
            } else if (s != ANYDIM) {
                dim_h = inputs[0]->sequence() * inputs[0]->head() / s;
                dim_s = s;
            } else {
                std::cout << "[TODO]Tensor.View not support!!!!" << std::endl;
            }
        } else if (b != -1 && h == -1 && s != -1 && d == -1) { // batch & sequence
            if (b != ANYDIM && s != ANYDIM) {
                assert(inputs[0]->sequence() * inputs[0]->batch() == b * s);
                dim_b = b;
                dim_s = s;
            } else if (b != ANYDIM) {
                dim_b = b;
                dim_s = inputs[0]->sequence() * inputs[0]->batch() / b;
            } else if (s != ANYDIM) {
                dim_b = inputs[0]->sequence() * inputs[0]->batch() / s;
                dim_s = s;
            } else {
                std::cout << "[TODO]Tensor.View not support!!!!" << std::endl;
            }
        } else {
            std::cout << "[TODO]Tensor.View not support!!!!" << std::endl;
        }
        outputs[0]->reshape(dim_b, dim_h, dim_s, dim_d);
        if ((b == -1 && s == -1 && inputs[0]->ctype() != BCTHW)                       // head & dimension
            || (b == 1 && h == 1 && inputs[0]->ctype() == BCTHW)                       // head & dimension
            || (b == 1 && h == 1 && inputs[0]->ctype() == BSHD)                       // head & dimension
            || (b == -1 && d == -1 && inputs[0]->ctype() == BSHD)                     // head & sequence
            || (h == -1 && d == -1 && inputs[0]->ctype() == BSHD)                     // batch & sequence
            || (b == -1 && h == 1 && s == 1 && d == -1 && inputs[0]->ctype() == BSHD) //  sequence & head & dimension -> dimension
            || (b == -1 && h == -1 && s == 1 && d == 1 && inputs[0]->ctype() == BSHD) //  sequence & head & dimension -> sequence
        ) {
            if (inputs[0]->masterTensor() == nullptr) {
                inputs[0]->free();
            }
            outputs[0]->setDtype(inputs[0]->dtype());
            outputs[0]->alloc();
            inputs[0]->shallowCopyFrom(outputs[0], false);
        } else {
            std::cout << "[TODO]Tensor.View alloc not support!!!!" << std::endl;
        }
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
    }
};

} // namespace mllm
#endif // CPUVIEWFUNC_HPP