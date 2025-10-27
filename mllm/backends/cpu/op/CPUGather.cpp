#include "CPUGather.hpp"
#include <vector>

namespace mllm {

CPUGather::CPUGather(Backend *bn, string opName, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
}

ErrorCode CPUGather::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // assert(inputs.size() == 3);
    assert(outputs.size() == 1);
    // if (inputs[1]->batch() == 0) {
    //     outputs[0]->reshape(inputs[0]->batch(), 1, inputs[0]->sequence(), inputs[0]->dimension());
    //     return Op::reshape(inputs, outputs);
    // }
    assert(inputs[0]->batch() == inputs[1]->batch());
    assert(inputs[0]->head() == inputs[1]->head());
    assert(inputs[0]->head() == 1);
    // assert(inputs[0]->dimension() == inputs[1]->dimension());
    // assert(inputs[1]->dimension() == 1);
    outputs[0]->reshape(inputs[0]->batch(), 1, inputs[0]->sequence(), inputs[1]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUGather::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (inputs[1]->batch() == 0) {
        return Op::execute(inputs, outputs);
    }

    assert(inputs[0]->ctype() == BSHD);
    assert(inputs[1]->ctype() == BSHD);
    assert(outputs[0]->ctype() == BSHD);
    auto input_indices = inputs[1];
    int hiddenSize = inputs[0]->dimension();
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
    for (int batch = 0; batch < inputs[0]->batch(); ++batch) {
        for (int seq = 0; seq < inputs[0]->sequence(); ++seq) {
            for (int indices = 0; indices < input_indices->dimension(); ++indices) {
                int dim_index = input_indices->dataAt<float>(batch, 0, seq, indices);
                float value = inputs[0]->dataAt<float>(batch, 0, seq, dim_index);
                outputs[0]->setDataAt<float>(batch, 0, seq, indices, value);
            }
        }
    }
    return Op::execute(inputs, outputs);
}

// ErrorCode CPUGather::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
//     if (inputs[0]->masterTensor() == nullptr) {
//         inputs[0]->free();
//     }
//     outputs[0]->setDtype(activation_dtype());
//     outputs[0]->alloc();
//     inputs[0]->shallowCopyFrom(outputs[0], false);
//     return MLLM_NO_ERROR;
// }
} // namespace mllm
