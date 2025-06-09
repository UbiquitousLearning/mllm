#include "backends/xnnpack/Functions/XpViewFunc.hpp"
#include "xnnpack.h"
#include <array>
#include "backends/xnnpack/Utils/Logger.hpp"

namespace mllm::xnnpack {

void XpViewFunction::reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    int b = (int)args[0];
    int h = (int)args[1];
    int s = (int)args[2];
    int d = (int)args[3];
    int dim_b = inputs[0]->batch();
    int dim_h = inputs[0]->head();
    int dim_s = inputs[0]->sequence();
    int dim_d = inputs[0]->dimension();
    if (b == -1 && h != -1 && s == -1 && d != -1) { // head & dimension
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
            Log::error("[TODO]Tensor.View not support!!!!");
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
            Log::error("[TODO]Tensor.View not support!!!!");
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
            Log::error("[TODO]Tensor.View not support!!!!");
        }
    } else {
        Log::error("[TODO]Tensor.View not support!!!!");
    }
    outputs[0]->reshape(dim_b, dim_h, dim_s, dim_d);
}

void XpViewFunction::execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    Log::warn("XpViewFunction will use reshape instead of view. Which will involve extra copy.");

    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return;

    int b = (int)args[0];
    int h = (int)args[1];
    int s = (int)args[2];
    int d = (int)args[3];
    int dim_b = inputs[0]->batch();
    int dim_h = inputs[0]->head();
    int dim_s = inputs[0]->sequence();
    int dim_d = inputs[0]->dimension();
    if (b == -1 && h != -1 && s == -1 && d != -1) { // head & dimension
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
            Log::error("[TODO]Tensor.View not support!!!!");
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
            Log::error("[TODO]Tensor.View not support!!!!");
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
            Log::error("[TODO]Tensor.View not support!!!!");
        }
    } else {
        Log::error("[TODO]Tensor.View not support!!!!");
    }

    std::array<size_t, 4> new_shape{(size_t)dim_b, (size_t)dim_s, (size_t)dim_h, (size_t)dim_d};
    auto status = xnn_define_static_reshape(xpb->getCurProcessingGraph()->getXnnSubgraph(), 4, new_shape.data(), inputs[0]->uuid(), outputs[0]->uuid(), 0);
    if (status != xnn_status_success) {
        Log::error("XpViewFunc xnn_define_static_reshape error");
    }
}
} // namespace mllm::xnnpack