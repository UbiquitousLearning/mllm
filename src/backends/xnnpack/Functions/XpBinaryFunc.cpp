#include "backends/xnnpack/Functions/XpBinaryFunc.hpp"
#include "Tensor.hpp"
#include "backends/xnnpack/XnnpackBackend.hpp"

namespace mllm::xnnpack {

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
        Log::error("XpAdd::setUp Error");
        exit(-1);
    }
}

} // namespace mllm::xnnpack