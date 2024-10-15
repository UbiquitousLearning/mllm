
#include "QNNReLU.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cassert>

namespace mllm {
QNNReLU::QNNReLU(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
        scale_.setBackend(bn);
}

ErrorCode QNNReLU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNReLU::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return graphAddNode(name(), "Relu", inputs, outputs, {}, "qti.aisw", true, &scale_);
}

ErrorCode QNNReLU::load(AbstructLoader &loader) {
    string scaleName = name();

    std::string wordToRemove = "relu";
    int pos = scaleName.find(wordToRemove);
    if (pos != -1) {
        scaleName.erase(pos, wordToRemove.length());
    }

    scale_.setName(scaleName + "input_scale");
    scale_.reshape(1, 1, 1, 1);
    scale_.setDtype(MLLM_TYPE_F32);
    scale_.alloc();
    loader.load(&scale_);

    std::cout <<  scale_.hostPtr<float>()[0] << std::endl;

    return Op::load(loader);
}

} // namespace mllm
