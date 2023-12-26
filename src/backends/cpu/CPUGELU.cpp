#include "CPUGELU.hpp"

#include <cmath>
#include <utility>

namespace mllm {
CPUGELU::CPUGELU(Backend *bn, string opName, bool multiThread):support_multi_thread_(multiThread), Op(bn, std::move(opName))  {
}

ErrorCode CPUGELU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUGELU::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    int batch = input->batch();
    int head = input->head();
    int seq = input->sequence();
    int dim = input->dimension();
#pragma omp parallel for collapse(4)
    for (int b = 0; b <batch ; ++b) {
        for (int h = 0; h < head; ++h) {
            for (int s = 0; s < seq; ++s) {
                for (int d = 0; d < dim; ++d) {
                    float value = input->dataAt<float>(b, h, s, d);
                    output->setDataAt<float>(b, h, s, d, 0.5 * value * (1 + std::tanh(std::sqrt(2 / M_PI) * (value + 0.044715 * std::pow(value, 3)))));
                }
            }
        }
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPUGELU::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}
} // namespace mllm