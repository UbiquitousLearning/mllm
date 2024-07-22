
#include "CPUPosition.hpp"

namespace mllm {

CPUPosition::CPUPosition(Backend *bn, string opName, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName){
}

ErrorCode CPUPosition::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];
    output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUPosition::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    int N = inputs[0]->batch();
    int C = inputs[0]->head();
    int H = inputs[0]->sequence();
    int W = inputs[0]->dimension();

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    outputs[0]->setDataAt<float>(n, c, h, w, h + pos_cnt_ +2);
                }
            }
        }
    }
    pos_cnt_ += H;

    return Op::execute(inputs, outputs);
}

ErrorCode CPUPosition::load(AbstructLoader &loader) {
    return Op::load(loader);
}

ErrorCode CPUPosition::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}

ErrorCode CPUPosition::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::setUp(inputs, outputs);
}
} // namespace mllm
