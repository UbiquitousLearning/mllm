#include "CPUGELU.hpp"

#include <cmath>
#include <utility>

namespace mllm {
CPUGELU::CPUGELU(Backend *bn, string opName, int threadCount):thread_count(threadCount), Op(bn, std::move(opName))  {
    if (!init_table_gelu_f16_flag) {
        init_table_gelu_f16();
        init_table_gelu_f16_flag = true;
    }
}

ErrorCode CPUGELU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
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
#pragma omp parallel for collapse(3) num_threads(thread_count)
    for (int b = 0; b <batch ; ++b) {
        for (int h = 0; h < head; ++h) {
            for (int s = 0; s < seq; ++s) {
//                for (int d = 0; d < dim; ++d) {
//                    float value = input->dataAt<float>(b, h, s, d);
//                    // output->setDataAt<float>(b, h, s, d, 0.5 * value * (1 + std::tanh(std::sqrt(2 / M_PI) * (value + 0.044715 * std::pow(value, 3)))));
//                    output->setDataAt<float>(b, h, s, d, 0.5 * value * (1 + std::tanh(std::sqrt(2 / M_PI) * (0.7978845608 * (value + 0.044715 * std::pow(value, 3))))));
//;
//                }
                mllm_vec_gelu_f32(dim,  outputs[0]->ptrAt<float>(b, h, s,0),
                            inputs[0]->ptrAt<float>(b, h, s,0));
            }
        }
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPUGELU::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}
} // namespace mllm