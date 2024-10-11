#include "backends/xnnpack/Functions/XpBinaryFunc.hpp"
#include "Tensor.hpp"
#include "backends/xnnpack/XnnpackBackend.hpp"
#include "xnnpack.h"

namespace mllm::xnnpack {

void XpBroadcastAddFunction::setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) {
    // reshape
    auto input = inputs[0];
    auto output = outputs[0];
    output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
    output->setDtype(input->dtype());
}

void XpBroadcastAddFunction::execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb, inputs);
    tryDefineAllXpTensors(xpb, outputs);

    // define xnnpack op.
    auto status = xnn_define_binary(
        xpb->getXnnSubgraph(),
        xnn_binary_add,
        nullptr,
        inputs[0]->uuid(),
        inputs[1]->uuid(),
        outputs[0]->uuid(),
        0);

    if (status != xnn_status_success) {
        Log::error("XpBroadcastAddFunction::execute Error");
        exit(-1);
    }
}

void XpBroadcastSubFunction::setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) {
    // reshape
    auto input = inputs[0];
    auto output = outputs[0];
    output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
    output->setDtype(input->dtype());
}

void XpBroadcastSubFunction::execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb, inputs);
    tryDefineAllXpTensors(xpb, outputs);

    // define xnnpack op.
    auto status = xnn_define_binary(
        xpb->getXnnSubgraph(),
        xnn_binary_subtract,
        nullptr,
        inputs[0]->uuid(),
        inputs[1]->uuid(),
        outputs[0]->uuid(),
        0);

    if (status != xnn_status_success) {
        Log::error("XpBroadcastSubFunction::execute Error");
        exit(-1);
    }
}

void XpBroadcastMulFunction::setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) {
    // reshape
    auto input = inputs[0];
    auto output = outputs[0];
    output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
    output->setDtype(input->dtype());
}

void XpBroadcastMulFunction::execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb, inputs);
    tryDefineAllXpTensors(xpb, outputs);

    // define xnnpack op.
    auto status = xnn_define_binary(
        xpb->getXnnSubgraph(),
        xnn_binary_multiply,
        nullptr,
        inputs[0]->uuid(),
        inputs[1]->uuid(),
        outputs[0]->uuid(),
        0);

    if (status != xnn_status_success) {
        Log::error("XpBroadcastMulFunction::execute Error");
        exit(-1);
    }
}

void XpBroadcastDivFunction::setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) {
    // reshape
    auto input = inputs[0];
    auto output = outputs[0];
    output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
    output->setDtype(input->dtype());
}

void XpBroadcastDivFunction::execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb, inputs);
    tryDefineAllXpTensors(xpb, outputs);

    // define xnnpack op.
    auto status = xnn_define_binary(
        xpb->getXnnSubgraph(),
        xnn_binary_divide,
        nullptr,
        inputs[0]->uuid(),
        inputs[1]->uuid(),
        outputs[0]->uuid(),
        0);

    if (status != xnn_status_success) {
        Log::error("XpBroadcastDivFunction::execute Error");
        exit(-1);
    }
}

void XpTTAddFunction::setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) {
    // reshape
    auto input = inputs[0];
    auto output = outputs[0];
    output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
    output->setDtype(input->dtype());
}

void XpTTAddFunction::execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb, inputs);
    tryDefineAllXpTensors(xpb, outputs);

    // define xnnpack op.
    auto status = xnn_define_binary(
        xpb->getXnnSubgraph(),
        xnn_binary_add,
        nullptr,
        inputs[0]->uuid(),
        inputs[1]->uuid(),
        outputs[0]->uuid(),
        0);

    if (status != xnn_status_success) {
        Log::error("XpTTAddFunc::execute Error");
        exit(-1);
    }
}

void XpTTSubFunction::setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) {
    // reshape
    auto input = inputs[0];
    auto output = outputs[0];
    output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
    output->setDtype(input->dtype());
}

void XpTTSubFunction::execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb, inputs);
    tryDefineAllXpTensors(xpb, outputs);

    // define xnnpack op.
    auto status = xnn_define_binary(
        xpb->getXnnSubgraph(),
        xnn_binary_subtract,
        nullptr,
        inputs[0]->uuid(),
        inputs[1]->uuid(),
        outputs[0]->uuid(),
        0);

    if (status != xnn_status_success) {
        Log::error("XpTTSubFunc::execute Error");
        exit(-1);
    }
}

void XpTTMulFunction::setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) {
    // reshape
    auto input = inputs[0];
    auto output = outputs[0];
    output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
    output->setDtype(input->dtype());
}

void XpTTMulFunction::execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb, inputs);
    tryDefineAllXpTensors(xpb, outputs);

    // define xnnpack op.
    auto status = xnn_define_binary(
        xpb->getXnnSubgraph(),
        xnn_binary_multiply,
        nullptr,
        inputs[0]->uuid(),
        inputs[1]->uuid(),
        outputs[0]->uuid(),
        0);

    if (status != xnn_status_success) {
        Log::error("XpTTMulFunc::execute Error");
        exit(-1);
    }
}

void XpTTDivFunction::setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) {
    // reshape
    auto input = inputs[0];
    auto output = outputs[0];
    output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
    output->setDtype(input->dtype());
}

void XpTTDivFunction::execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb, inputs);
    tryDefineAllXpTensors(xpb, outputs);

    // define xnnpack op.
    auto status = xnn_define_binary(
        xpb->getXnnSubgraph(),
        xnn_binary_divide,
        nullptr,
        inputs[0]->uuid(),
        inputs[1]->uuid(),
        outputs[0]->uuid(),
        0);

    if (status != xnn_status_success) {
        Log::error("XpTTDivFunc::execute Error");
        exit(-1);
    }
}

} // namespace mllm::xnnpack