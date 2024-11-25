
#include "CPUAdd.hpp"
#include "../compute/Arithmetic.hpp"

namespace mllm {

CPUAdd::CPUAdd(Backend *bn, string opName, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
}

ErrorCode CPUAdd::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);
    if (inputs[0]->batch() == 1 || inputs[1]->batch() == 1) {
    } else {
        assert(inputs[0]->batch() == inputs[1]->batch());
    }
    assert(inputs[0]->head() == inputs[1]->head());
    assert(inputs[0]->sequence() == inputs[1]->sequence());
    assert(inputs[0]->dimension() == inputs[1]->dimension());
    outputs[0]->reshape(std::max(inputs[0]->batch(), inputs[1]->batch()), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    // outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUAdd::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input0 = inputs[0];
    auto input1 = inputs[1];
    int batch_ = std::max(input0->batch(), input1->batch());
    for (int n = 0; n < batch_; ++n) {
        auto n_0 = std::min(n, input0->batch() - 1);
        auto n_1 = std::min(n, input1->batch() - 1);
#pragma omp parallel for collapse(2) num_threads(CPUBackend::cpu_threads)
        for (int c = 0; c < input0->head(); ++c) {
            for (int h = 0; h < input0->sequence(); ++h) {
                mllm_add_fp32(input0->ptrAt<float>(n_0, c, h, 0), input1->ptrAt<float>(n_0, c, h, 0),
                              outputs[0]->ptrAt<float>(n_0, c, h, 0), input0->dimension());
            }
        }
    }
    return Op::execute(inputs, outputs);
    /*
        int N = std::max(inputs[0]->batch(), inputs[1]->batch());
        int C = inputs[0]->head();
        int H = inputs[0]->sequence();
        int W = inputs[0]->dimension();
        for (int n = 0; n < N; ++n) {
            auto n_0 = std::min(n, inputs[0]->batch() - 1);
            auto n_1 = std::min(n, inputs[1]->batch() - 1);
            if (inputs[0]->masterTensor() == nullptr && inputs[1]->masterTensor() == nullptr && inputs[0]->ctype() == inputs[1]->ctype()) {
                auto copy_size = C * H * W;
                auto in0_ptr = inputs[0]->ptrAt<float>(n_0, 0, 0, 0);
                auto in1_ptr = inputs[1]->ptrAt<float>(n_1, 0, 0, 0);
                auto out_ptr = outputs[0]->ptrAt<float>(n, 0, 0, 0);
    #pragma omp parallel for num_threads(thread_count)
                for (int is = 0; is < copy_size; ++is) {
                    out_ptr[is] = in0_ptr[is] + in1_ptr[is];
                }
            } else {
                for (int c = 0; c < C; ++c) {
                    for (int h = 0; h < H; ++h) {
    #pragma omp parallel for num_threads(thread_count)
                        for (int w = 0; w < W; ++w) {
                            outputs[0]->setDataAt<float>(n, c, h, w, inputs[0]->dataAt<float>(n_0, c, h, w) + inputs[1]->dataAt<float>(n_1, c, h, w));
                        }
                    }
                }
            }
        }
        return Op::execute(inputs, outputs);
        */
}
} // namespace mllm
