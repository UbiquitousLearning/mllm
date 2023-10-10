#include "CPUEmbedding.hpp"
mllm::CPUEmbedding::CPUEmbedding(mllm::Backend *bn, int hiddenSize, int vocabSize) :
    Op(bn), hiddenSize_(hiddenSize), vocabSize_(vocabSize) {
    CHECK_GT(hiddenSize_, 0);
    CHECK_GT(vocabSize_, 0);
    weight_.setBackend(bn);
}
ErrorCode mllm::CPUEmbedding::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout<<name() << "  CPUEmbedding  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    auto input = inputs[0];
    auto output = outputs[0];
    CHECK_EQ(input->width(), 1);
    output->reshape(input->batch(), input->sequence(), hiddenSize_, 1);
    weight_.reshape(vocabSize_, hiddenSize_, 1, 1);
    return NO_ERROR;
}
ErrorCode mllm::CPUEmbedding::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout<<name() << "  CPUEmbedding  setUp" << std::endl;

    return Op::setUp(inputs, outputs);
}
ErrorCode mllm::CPUEmbedding::load(mllm::ParamLoader &loader) {
    return Op::load(loader);
}
ErrorCode mllm::CPUEmbedding::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout<<name() << "  CPUEmbedding  execute" << std::endl;
    return NO_ERROR;
}
