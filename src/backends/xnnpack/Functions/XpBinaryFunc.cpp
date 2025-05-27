#include "backends/xnnpack/Functions/XpBinaryFunc.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "backends/xnnpack/XnnpackBackend.hpp"
#include "xnnpack.h"

namespace mllm::xnnpack {

void XpBroadcastAddFunction::reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    // reshape
    auto input = inputs[0];
    auto output = outputs[0];
    output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
    output->setDtype(input->dtype());
}

void XpBroadcastAddFunction::execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return;

    // TODO memory leak;
    Tensor constant_v;
    constant_v.setBackend(xpb);
    constant_v.reshape(1, 1, 1, 1);
    constant_v.setDtype(DataType::MLLM_TYPE_F32);
    constant_v.alloc();
    constant_v.setDataAt(0, 0, 0, 0, (float)args[0]);
    defineWeightTensor(xpb->getCurProcessingGraph(), &constant_v);

    // define xnnpack op.
    auto status = xnn_define_binary(
        xpb->getCurProcessingGraph()->getXnnSubgraph(),
        xnn_binary_add,
        nullptr,
        inputs[0]->uuid(),
        constant_v.uuid(),
        outputs[0]->uuid(),
        0);

    if (status != xnn_status_success) {
        Log::error("XpBroadcastAddFunction::execute Error");
        exit(-1);
    }
}

void XpBroadcastSubFunction::reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    // reshape
    auto input = inputs[0];
    auto output = outputs[0];
    output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
    output->setDtype(input->dtype());
}

void XpBroadcastSubFunction::execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return;

    // TODO memory leak;
    Tensor constant_v;
    constant_v.setBackend(xpb);
    constant_v.reshape(1, 1, 1, 1);
    constant_v.setDtype(DataType::MLLM_TYPE_F32);
    constant_v.alloc();
    constant_v.setDataAt(0, 0, 0, 0, (float)args[0]);
    defineWeightTensor(xpb->getCurProcessingGraph(), &constant_v);

    // define xnnpack op.
    auto status = xnn_define_binary(
        xpb->getCurProcessingGraph()->getXnnSubgraph(),
        xnn_binary_subtract,
        nullptr,
        inputs[0]->uuid(),
        constant_v.uuid(),
        outputs[0]->uuid(),
        0);

    if (status != xnn_status_success) {
        Log::error("XpBroadcastSubFunction::execute Error");
        exit(-1);
    }
}

void XpBroadcastMulFunction::reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    // reshape
    auto input = inputs[0];
    auto output = outputs[0];
    output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
    output->setDtype(input->dtype());
}

void XpBroadcastMulFunction::execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return;

    // TODO memory leak;
    Tensor constant_v;
    constant_v.setBackend(xpb);
    constant_v.reshape(1, 1, 1, 1);
    constant_v.setDtype(DataType::MLLM_TYPE_F32);
    constant_v.alloc();
    constant_v.setDataAt(0, 0, 0, 0, (float)args[0]);
    defineWeightTensor(xpb->getCurProcessingGraph(), &constant_v);

    // define xnnpack op.
    auto status = xnn_define_binary(
        xpb->getCurProcessingGraph()->getXnnSubgraph(),
        xnn_binary_multiply,
        nullptr,
        inputs[0]->uuid(),
        constant_v.uuid(),
        outputs[0]->uuid(),
        0);

    if (status != xnn_status_success) {
        Log::error("XpBroadcastMulFunction::execute Error");
        exit(-1);
    }
}

void XpBroadcastDivFunction::setup(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    // reshape
    auto input = inputs[0];
    auto output = outputs[0];
    output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
    output->setDtype(input->dtype());
}

void XpBroadcastDivFunction::execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return;

    // TODO memory leak;
    Tensor constant_v;
    constant_v.setBackend(xpb);
    constant_v.reshape(1, 1, 1, 1);
    constant_v.setDtype(DataType::MLLM_TYPE_F32);
    constant_v.alloc();
    constant_v.setDataAt(0, 0, 0, 0, (float)args[0]);
    defineWeightTensor(xpb->getCurProcessingGraph(), &constant_v);

    // define xnnpack op.
    auto status = xnn_define_binary(
        xpb->getCurProcessingGraph()->getXnnSubgraph(),
        xnn_binary_divide,
        nullptr,
        inputs[0]->uuid(),
        constant_v.uuid(),
        outputs[0]->uuid(),
        0);

    if (status != xnn_status_success) {
        Log::error("XpBroadcastDivFunction::execute Error");
        exit(-1);
    }
}

void XpTTAddFunction::setup(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    // reshape
    auto input = inputs[0];
    auto output = outputs[0];
    output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
    output->setDtype(input->dtype());
}

void XpTTAddFunction::execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return;

    // define xnnpack op.
    auto status = xnn_define_binary(
        xpb->getCurProcessingGraph()->getXnnSubgraph(),
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

void XpTTSubFunction::setup(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    // reshape
    auto input = inputs[0];
    auto output = outputs[0];
    output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
    output->setDtype(input->dtype());
}

void XpTTSubFunction::execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return;

    // define xnnpack op.
    auto status = xnn_define_binary(
        xpb->getCurProcessingGraph()->getXnnSubgraph(),
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

void XpTTMulFunction::setup(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    // reshape
    auto input = inputs[0];
    auto output = outputs[0];
    output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
    output->setDtype(input->dtype());
}

void XpTTMulFunction::execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return;

    // define xnnpack op.
    auto status = xnn_define_binary(
        xpb->getCurProcessingGraph()->getXnnSubgraph(),
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

void XpTTDivFunction::setup(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    // reshape
    auto input = inputs[0];
    auto output = outputs[0];
    output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
    output->setDtype(input->dtype());
}

void XpTTDivFunction::execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return;

    // define xnnpack op.
    auto status = xnn_define_binary(
        xpb->getCurProcessingGraph()->getXnnSubgraph(),
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