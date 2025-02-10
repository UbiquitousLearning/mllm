
#include "CPUCausalMask.hpp"
#include <cmath>

namespace mllm {


CPUCausalMask::CPUCausalMask(Backend *bn, string opName, int threadCount) : thread_count(threadCount),
    Op(bn, opName) {
}

ErrorCode CPUCausalMask::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout << "CPUMask  reshape" << std::endl;
    // assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUCausalMask::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if(inputs[0]->sequence() >1 ) {
        int batch_size = inputs[0]->batch();
        int head_num = inputs[0]->head();
        int sequence = inputs[0]->sequence();
        int dimension = inputs[0]->dimension();
        // memset(outputs[0]->hostPtr<float>(),-INFINITY,outputs[0]->count() * sizeof(float));
        int old_dim = 0;
        if (inputs.size()>1) {
            old_dim = (int)inputs[1]->dataAt<float>(0,0,0,0)-sequence;
        }else{
#ifndef LLAMAFILE_SGEMM
            old_dim = dimension - sequence;
#endif
        }
#pragma omp parallel for collapse(4) num_threads(thread_count)
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < head_num; ++h) {
                for (int s = 0; s < sequence; ++s) {
                    for (int d = 0; d < inputs[0]->dimension(); ++d) {
                        if (d > s + old_dim) {
                            outputs[0]->setDataAt<float>({n, h, s, d}, -INFINITY);
                        }
                        else{
                            outputs[0]->setDataAt<float>({n, h, s, d}, inputs[0]->dataAt<float>(n, h, s, d));
                        }
                    }
                }
            }
        }
    }
    else{
        outputs[0]->copyFrom(inputs[0]);
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPUCausalMask::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    if(inputs[0]->masterTensor() == nullptr) {
        inputs[0]->free(); // TODO remove
    }
    outputs[0]->setDtype(activation_dtype());
    outputs[0]->alloc();
    inputs[0]->shallowCopyFrom(outputs[0].get(), false);
    return MLLM_NO_ERROR;
}
} // namespace mllm
