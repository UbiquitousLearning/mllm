#include <cmath>
#include "CPURMSNorm.hpp"
#include "Tensor.hpp"

namespace mllm {

// template class CPURMSNorm;
// template class CPURMSNorm;
// int32_t opp = 897988541;

int32_t op_params[1];
CPURMSNorm::CPURMSNorm(Backend *bn, string opName, bool multiThread, float epsilon) :
    Op(bn, opName), epsilon_(epsilon), support_multi_thread_(multiThread) {
    op_params[0] = 897988541;
    memcpy(&epsilon_, op_params, sizeof(float));
    weight_.setBackend(bn);
}

ErrorCode CPURMSNorm::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // RMSNorm 类似于LayerNorm作用于channel维度
    normSize_ = inputs[0]->dimension();
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    // outputs[0]->setDtype(activationDtype());
    // std::cout << name() << "  CPURMSNorm  reshape" << std::endl;
    return Op::reshape(inputs, outputs);
}

ErrorCode CPURMSNorm::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    int batch = input->batch();
    int dim = input->dimension();
    int seq = input->sequence();
    int head = input->head();
    for (int h = 0; h < head; h++) {
        for (int n = 0; n < batch; n++) {
            for (int s = 0; s < seq; s++) {
                double sum_squares = 0.0F;
                // sum
                // #pragma omp parallel for reduction(+ : sum_squares) num_threads(4)
                for (int d = 0; d < dim; d++) {
                    float value = input->dataAt<float>(n, h, s, d);
                    sum_squares += (double)value * value;
                }
                const float mean = sum_squares/dim;
                const float rms = 1.0f/sqrtf(mean + epsilon_);
                // use memset to set the value of the memory block
                #pragma omp parallel for num_threads(4)
                for (int d = 0; d < dim; d++) {
                    float value = input->dataAt<float>(n, h, s, d);
                    outputs[0]->setDataAt<float>(n, h, s, d, weight_.dataAt<float>(0, 0, 0, d) * value * rms);
                }
            }
        }
    }
    //    input->printData<float>();
    //    weight_.printData<float>();
    //    outputs[0]->printData<float>();

    // std::cout << name() << "  CPURMSNorm()" << std::endl;
    return Op::execute(inputs, outputs);
}
ErrorCode CPURMSNorm::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, 1, normSize_); //
    if (&loader != nullptr) {
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        loader.load(&weight_);
    } else {
        weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
    }
    return Op::load(loader);
}
ErrorCode CPURMSNorm::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    return Op::free(inputs, outputs);
}
} // namespace mllm