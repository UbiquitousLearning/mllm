#include "CPUGather.hpp"
#include <vector>

namespace mllm {

CPUGather::CPUGather(Backend *bn,  string opName, bool multiThread) :
    Op(bn, opName) {
}

ErrorCode CPUGather::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUGather  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 3);
    CHECK_EQ(outputs.size(), 1);
    if(inputs[1]->batch() == 0) {
        outputs[0]->reshape(inputs[0]->batch(), 1, inputs[0]->sequence(), inputs[0]->dimension());
        return Op::reshape(inputs, outputs);
    }
    CHECK_EQ(inputs[0]->batch(), inputs[1]->batch());
    CHECK_EQ(inputs[0]->head(), inputs[1]->head());
    CHECK_EQ(inputs[0]->head(), 1);
    CHECK_EQ(inputs[0]->dimension(), inputs[1]->dimension());
    CHECK_EQ(inputs[0]->sequence(), inputs[2]->sequence());
    CHECK_EQ(inputs[2]->dimension(), 1);
    outputs[0]->reshape(inputs[0]->batch(), 1, inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUGather::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if(inputs[1]->batch() == 0) {
        return Op::execute(inputs, outputs);
    }
    //std::cout<<name() << "  CPUGather()" << std::endl;
    assert(inputs[0]->ctype() == BSHD);
    assert(inputs[1]->ctype() == BSHD);
    assert(outputs[0]->ctype() == BSHD);
    auto input_indices = inputs[2];
    int hiddenSize = inputs[0]->dimension();
    for (int batch = 0; batch < inputs[0]->batch(); ++batch) {
        for (int seq = 0; seq < inputs[0]->sequence(); ++seq) {
            if(input_indices->dataAt<float>(batch, 0, seq, 0) >= 0) {
                memcpy(outputs[0]->hostPtr<float>() + outputs[0]->offset(batch, 0, seq, 0),
                       inputs[1]->hostPtr<float>() + (int)inputs[1]->offset(batch, 0, input_indices->dataAt<float>(batch, 0, seq, 0), 0),
                       inputs[1]->dtypeSize() * hiddenSize);
            }
        }
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPUGather::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUGather() setUp" << std::endl;
    if(inputs[0]->masterTensor() == nullptr) {
        inputs[0]->free();
    }
    outputs[0]->setDtype(activation_dtype());
    outputs[0]->alloc();
    inputs[0]->deepCopyFrom(outputs[0], false);
    return NO_ERROR;
}
} // namespace mllm

