
#include "CPUSiLU.hpp"
#include <cmath>
#include "../compute/ActivationFunction.hpp"

namespace mllm {

CPUSiLU::CPUSiLU(Backend *bn, string opName, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    if (!init_table_silu_f16_flag) {
        init_table_silu_f16();
        init_table_silu_f16_flag = true;
    }
}

ErrorCode CPUSiLU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    // outputs[0]->setDtype(activationDtype());

    return Op::reshape(inputs, outputs);
}

ErrorCode CPUSiLU::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (inputs[0]->sequence() != outputs[0]->sequence() && outputs[0]->masterTensor() == nullptr) {
        outputs[0]->reshape(outputs[0]->batch(), outputs[0]->head(), inputs[0]->sequence(), outputs[0]->dimension());
        // outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->alloc();
    }

    auto input = inputs[0];
    int batch = input->batch();
    int n1 = input->head();
    int n2 = input->sequence();
    int n3 = input->dimension();
#pragma omp parallel for collapse(3) num_threads(thread_count)
    for (int n = 0; n < batch; n++) {
        for (int h = 0; h < n2; h++) {
            for (int c = 0; c < n1; c++) {
                //                #pragma omp parallel for num_threads(thread_count)
                //                for (int w = 0; w < n3; w++) {
                //                    float value = input->dataAt<float>(n, c, h, w);
                //                    outputs[0]->setDataAt<float>(n, c, h, w, value / (1 + std::exp(-value)));
                //                }
                mllm_vec_silu_f32(n3, outputs[0]->ptrAt<float>(n, c, h, 0),
                                  inputs[0]->ptrAt<float>(n, c, h, 0));
            }
        }
    }

    return Op::execute(inputs, outputs);
}

} // namespace mllm
