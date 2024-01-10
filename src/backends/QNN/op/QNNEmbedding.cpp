
#include "QNNEmbedding.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNEmbedding::QNNEmbedding(Backend *bn, string opName, int hiddenSize, int vocabSize) :
    QNNCommonOp(bn, opName), hiddenSize_(hiddenSize), vocabSize_(vocabSize) {
    CHECK_GT(hiddenSize_, 0);
    CHECK_GT(vocabSize_, 0);
    weight_.setBackend(bn);
}

ErrorCode QNNEmbedding::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    auto input = inputs[0];
    auto output = outputs[0];
    // Input: [batch, 1, sequence, 1]
    output->reshape(input->batch(), 1, input->sequence(), hiddenSize_);
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNEmbedding::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return graphAddNode(name(), "Add", inputs, outputs);
}

ErrorCode QNNEmbedding::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, vocabSize_, hiddenSize_);
    weight_.setDtype(loader.getDataType(weight_.name()));
    weight_.alloc();
    loader.load(&weight_);
    return Op::load(loader);
}

ErrorCode QNNEmbedding::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    return Op::free(inputs, outputs);
}

} // namespace mllm

