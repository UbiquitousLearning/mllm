
#include "CPUQuickGELU.hpp"

namespace mllm {

CPUQuickGELU::CPUQuickGELU(Backend *bn,  string opName, int threadCount) : thread_count(threadCount),
    Op(bn, opName) {
    if (!init_table_gelu_quick_f16_flag) {
        init_table_gelu_quick_f16();
        init_table_gelu_quick_f16_flag = true;
    }
}

ErrorCode CPUQuickGELU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}


ErrorCode CPUQuickGELU::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
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
//                    output->setDataAt<float>(b, h, s, d, value * (1 / (1 + std::exp(-1.702 * value))));
//                }
                mllm_vec_gelu_quick_f32(dim,  outputs[0]->ptrAt<float>(b, h, s,0),
                                  inputs[0]->ptrAt<float>(b, h, s,0));
            }
        }
    }
    return Op::execute(inputs, outputs);
}

} // namespace mllm

