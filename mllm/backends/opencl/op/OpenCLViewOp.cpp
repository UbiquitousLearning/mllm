#include "OpenCLViewOp.hpp"
#include "Types.hpp"
// #include "utils/OpenCLTools.hpp"
#include <cassert>
#include <iostream>

namespace mllm {

OpenCLViewOp::OpenCLViewOp(Backend *bn, std::string name, int b, int h, int s, int d) :
    Op(bn, std::move(name)), b(b), h(h), s(s), d(d) {
}

OpenCLViewOp::~OpenCLViewOp() {
}

ErrorCode OpenCLViewOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
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
        dim_s = inputs[0]->channel() * inputs[0]->time() * inputs[0]->batch() * inputs[0]->height() * inputs[0]->width() / d;
        dim_h = 1;
        dim_d = d;
    } else if (b == 1 && h == 1 && s == -1 && d != -1) { //  batch & head & sequence -> sequence
        dim_b = 1;
        dim_s = inputs[0]->sequence() * inputs[0]->batch() * inputs[0]->head() * inputs[0]->dimension() / d;
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
    if (inputs[0]->ctype() == BCTHW && inputs[0]->name() == outputs[0]->name()) {
        outputs[0]->setCtype(BSHD);
    }
    outputs[0]->reshape(dim_b, dim_h, dim_s, dim_d);
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLViewOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs[0]->backend()->type() == MLLM_OPENCL);
    if (inputs[0]->sequence() > 0) {
        assert(inputs[0] == outputs[0]);
    }
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLViewOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    return MLLM_NO_ERROR;
}

} // namespace mllm