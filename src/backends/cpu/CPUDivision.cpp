
#include "CPUDivision.hpp"

namespace mllm {

CPUDivision::CPUDivision(Backend *bn, string opName, int threadCount) : thread_count(threadCount),
    Op(bn, opName) {
}

ErrorCode CPUDivision::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    assert(inputs.size() == 2);
    assert(outputs.size() == 1);
    if (inputs[1]->count() != 1) {
        assert(inputs[0]->batch() == inputs[1]->batch());
        assert(inputs[0]->head() == inputs[1]->head());
        assert(inputs[0]->sequence() == inputs[1]->sequence());
        assert(inputs[0]->dimension() == inputs[1]->dimension());
    }
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());

    // outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}
ErrorCode CPUDivision::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    int N = inputs[0]->batch();
    int C = inputs[0]->head();
    int H = inputs[0]->sequence();
    int W = inputs[0]->dimension();
    int in1_000 = inputs[1]->dataAt<float>(0, 0, 0, 0);
    if(inputs[0]->masterTensor() == nullptr && inputs[1]->masterTensor() == nullptr && inputs[0]->ctype() == inputs[1]->ctype()) {
        auto copy_size = N * C * H * W;
        auto in0_ptr = inputs[0]->hostPtr<float>();
        auto in1_ptr = inputs[1]->hostPtr<float>();
        auto out_ptr = outputs[0]->hostPtr<float>();
#pragma omp parallel for num_threads(thread_count)
        for (int is = 0; is < copy_size; ++is) {
            if (inputs[1]->count() == 1) {
                out_ptr[is] = in0_ptr[is] / in1_000;
            }else {
                out_ptr[is] = in0_ptr[is] / in1_ptr[is];
            }
        }
    }else {
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
#pragma omp parallel for num_threads(thread_count)
                    for (int w = 0; w < W; ++w) {
                        auto divisor = (inputs[1]->count() != 1) ?
                                           inputs[1]->dataAt<float>(n, c, h, w) :
                                           inputs[1]->dataAt<float>(0, 0, 0, 0);
                        outputs[0]->setDataAt<float>(n, c, h, w,
                                                     inputs[0]->dataAt<float>(n, c, h, w) / divisor);
                    }
                }
            }
        }
    }
    return Op::execute(inputs, outputs);
}

} // namespace mllm
