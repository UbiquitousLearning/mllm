
#include "CPUMergeOutput.hpp"
#include <cstring>

namespace mllm {

CPUMergeOutput::CPUMergeOutput(Backend *bn, string opName, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
}

ErrorCode CPUMergeOutput::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);

    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() + inputs[1]->sequence() * 4, inputs[0]->dimension());

    return Op::reshape(inputs, outputs);
}

ErrorCode CPUMergeOutput::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    activation_dtype_ = inputs[0]->dtype();
    return Op::setUp(inputs, outputs);
}

ErrorCode CPUMergeOutput::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // copy data from input to output
    int offset = 0;
    memcpy(outputs[0]->hostPtr<int8_t>(), inputs[0]->hostPtr<uint8_t>(), inputs[0]->cntSize());
    offset += inputs[0]->cntSize();
    memcpy(outputs[0]->hostPtr<int8_t>() + offset, inputs[1]->hostPtr<uint8_t>(), inputs[1]->cntSize());
    

    return Op::execute(inputs, outputs);
}
} // namespace mllm

