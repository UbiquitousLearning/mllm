/**
 * @file CPUSlidingWindowMask.cpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-04-27
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "CPUSlidingWindowMask.hpp"
#include <limits>

namespace mllm {

CPUSlidingWindowMask::CPUSlidingWindowMask(Backend *bn, string opName, int windowSize, int threadCount) :
    thread_count(threadCount),
    winodw_size(windowSize),
    Op(bn, opName) {
}

ErrorCode CPUSlidingWindowMask::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUSlidingWindowMask::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (inputs[0]->sequence() > 1) {
        int batch_size = inputs[0]->batch();
        int head_num = inputs[0]->head();
        int sequence = inputs[0]->sequence();
        int dimension = inputs[0]->dimension();
        int old_dim = dimension - sequence;
        int _t_window_size = winodw_size - 1;
        for (int n = 0; n < batch_size; ++n) {
            for (int h = 0; h < head_num; ++h) {
                for (int s = 0; s < sequence; ++s) {
#pragma omp parallel for num_threads(thread_count)
                    for (int d = 0; d < dimension; ++d) {
                        if (/*right bound of window*/ d > s + old_dim || /*left bound of window*/ d < s - _t_window_size) {
                            outputs[0]->setDataAt<float>({n, h, s, d}, std::numeric_limits<float>::lowest());
                        } else {
                            outputs[0]->setDataAt<float>({n, h, s, d}, inputs[0]->dataAt<float>(n, h, s, d));
                        }
                    }
                }
            }
        }
    } else {
        outputs[0]->copyFrom(inputs[0]);
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPUSlidingWindowMask::load(AbstructLoader &loader) {
    return Op::load(loader);
}

ErrorCode CPUSlidingWindowMask::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}

ErrorCode CPUSlidingWindowMask::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    if (inputs[0]->masterTensor() == nullptr) {
        inputs[0]->free(); // TODO remove
    }
    outputs[0]->setDtype(activation_dtype());
    outputs[0]->alloc();
    inputs[0]->shallowCopyFrom(outputs[0].get(), false);
    return MLLM_NO_ERROR;
}
} // namespace mllm
