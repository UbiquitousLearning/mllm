
#include "QNNGELU.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNGELU::QNNGELU(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
        scale_.setBackend(bn);
}

ErrorCode QNNGELU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNGELU::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //Todo: gelu do not supprt signed fix int8
    return graphAddNode(name(), "Gelu", inputs, outputs, {}, "qti.aisw", true, &scale_);
}

ErrorCode QNNGELU::load(AbstructLoader &loader) {
    string scaleName = name();

    std::string wordToRemove = "gelu";
    int pos = scaleName.find(wordToRemove);
    if (pos != -1) {
        scaleName.erase(pos, wordToRemove.length());
    }

    scale_.setName(scaleName + "input_scale");
    scale_.reshape(1, 1, 1, 1);
    scale_.setDtype(MLLM_TYPE_F32);
    scale_.alloc();
    loader.load(&scale_);

    return Op::load(loader);
}

} // namespace mllm

