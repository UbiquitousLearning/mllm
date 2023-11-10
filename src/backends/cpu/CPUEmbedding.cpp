#include "CPUEmbedding.hpp"
#include "ParamLoader.hpp"
mllm::CPUEmbedding::CPUEmbedding(mllm::Backend *bn,  string opName, int hiddenSize, int vocabSize) :
    Op(bn, opName), hiddenSize_(hiddenSize), vocabSize_(vocabSize) {
    CHECK_GT(hiddenSize_, 0);
    CHECK_GT(vocabSize_, 0);
    weight_.setBackend(bn);
}
ErrorCode mllm::CPUEmbedding::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUEmbedding  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    auto input = inputs[0];
    auto output = outputs[0];
    // Input: [batch, 1, sequence, 1]
//    CHECK_EQ(input->width(), 1);
    output->reshape(input->batch(), 1, input->sequence(), hiddenSize_);
    //outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}

ErrorCode mllm::CPUEmbedding::load(mllm::AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, vocabSize_, hiddenSize_);
    weight_.setDtype(loader.getDataType(weight_.name()));
    weight_.alloc();
    loader.load(&weight_);
    return Op::load(loader);
}
ErrorCode mllm::CPUEmbedding::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUEmbedding  execute" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    auto input = inputs[0];
    auto output = outputs[0];
    for (int batch = 0; batch < input->batch(); ++batch) {
        for (int head = 0; head < input->head(); ++head) {
            #pragma omp parallel for num_threads(8)
            for (int seq = 0; seq < input->sequence(); ++seq) {
                //                std::cout<<"batch: "<<batch<<" channel: "<<channel<<" seq: "<<seq<<std::endl;
                //                std::cout<<"input->dataAt<int>(batch, channel, seq, 0): "<<input->dataAt<int>(batch, channel, seq, 0)<<std::endl;

                // Set the seq
                memcpy(output->hostPtr<float>() + output->offset(batch, head, seq, 0),
                       weight_.hostPtr<float>() + weight_.offset(0, 0, (int)input->dataAt<float>(batch, head, seq, 0), 0),
                       weight_.dtypeSize() * hiddenSize_);
            }
        }
    }
    //    output->printData<float>();
    return NO_ERROR;
}
ErrorCode mllm::CPUEmbedding::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    return Op::free(inputs, outputs);
}
