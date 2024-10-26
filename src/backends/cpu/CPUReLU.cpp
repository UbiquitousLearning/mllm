//
// Created by Xiang Li on 2023/11/26.
//

#include "CPUReLU.hpp"

#include <utility>

namespace mllm {
CPUReLU::CPUReLU(Backend *bn, string opName, int threadCount) :
    thread_count(threadCount), Op(bn, std::move(opName)) {
}
ErrorCode CPUReLU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUReLU::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    int batch = input->batch();
    int head = input->head();
    int seq = input->sequence();
    int dim = input->dimension();
#pragma omp parallel for collapse(4) num_threads(thread_count)
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < head; ++h) {
            for (int s = 0; s < seq; ++s) {
                for (int d = 0; d < dim; ++d) {
                    float value = input->dataAt<float>(b, h, s, d);
                    output->setDataAt<float>(b, h, s, d, value > 0 ? value : 0);
                }
            }
        }
    }
    return Op::execute(inputs, outputs);
}
ErrorCode CPUReLU::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}
} // namespace mllm