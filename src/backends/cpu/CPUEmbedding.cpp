#include "CPUEmbedding.hpp"
mllm::CPUEmbedding::CPUEmbedding(mllm::Backend *bn,  string opName, int hiddenSize, int vocabSize) :
    Op(bn, opName), hiddenSize_(hiddenSize), vocabSize_(vocabSize) {
    CHECK_GT(hiddenSize_, 0);
    CHECK_GT(vocabSize_, 0);
    weight_.setBackend(bn);
}
ErrorCode mllm::CPUEmbedding::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout<<name() << "  CPUEmbedding  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    auto input = inputs[0];
    auto output = outputs[0];
    // Input: [batch, 1, sequence, 1]
//    CHECK_EQ(input->width(), 1);
    output->reshape(input->batch(), 1, input->sequence(), hiddenSize_);
    weight_.reshape(1, 1, vocabSize_, hiddenSize_);
    weight_.setName(name() + ".weight");
    return NO_ERROR;
}
ErrorCode mllm::CPUEmbedding::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout<<name() << "  CPUEmbedding  setUp" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    if (!inputs[0]->allocted()) {
        inputs[0]->alloc();
    }
    outputs[0]->alloc();
    weight_.alloc();
//        inputs[0]->fullData<float>(1);
//        weight_.fullDataTest();
    //    inputs[0]->printData<int>();
    //    weight_.printData<float>();
    return NO_ERROR;
}
ErrorCode mllm::CPUEmbedding::load(mllm::ParamLoader &loader) {
    loader.load(&weight_);
    return NO_ERROR;
}
ErrorCode mllm::CPUEmbedding::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout<<name() << "  CPUEmbedding  execute" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    auto input = inputs[0];
    auto output = outputs[0];
    for (int batch = 0; batch < input->batch(); ++batch) {
        for (int head = 0; head < input->head(); ++head) {
            for (int seq = 0; seq < input->sequence(); ++seq) {
                //                std::cout<<"batch: "<<batch<<" channel: "<<channel<<" seq: "<<seq<<std::endl;
                //                std::cout<<"input->dataAt<int>(batch, channel, seq, 0): "<<input->dataAt<int>(batch, channel, seq, 0)<<std::endl;

                // Set the seq
                memcpy(output->hostPtr<float>() + output->offset(batch, head, seq, 0),
                       weight_.hostPtr<float>() + weight_.offset(0, 0, (int)input->dataAt<float>(batch, head, seq, 0), 0),
                       weight_.byteWidth() * hiddenSize_);
            }
        }
    }
    //    output->printData<float>();
    return NO_ERROR;
}
ErrorCode mllm::CPUEmbedding::reshapeOutputs(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->reshape(inputs[0]->batch(), 1, inputs[0]->sequence(), hiddenSize_);
    outputs[0]->alloc();
    return NO_ERROR;
}
ErrorCode mllm::CPUEmbedding::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    return Op::free(inputs, outputs);
}
