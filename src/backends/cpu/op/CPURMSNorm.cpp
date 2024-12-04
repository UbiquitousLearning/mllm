#include <cmath>
#include "CPURMSNorm.hpp"
#include "Tensor.hpp"
#include "Timing.hpp"
#include "../compute/VecDot.hpp"

namespace mllm {

// int32_t opp = 897988541;

// int32_t op_params[1];
CPURMSNorm::CPURMSNorm(Backend *bn, string opName, int normSize, float epsilon, bool add_unit_offset_, int threadCount) :
    thread_count(threadCount), add_unit_offset_(add_unit_offset_),
    Op(bn, opName), epsilon_(epsilon) {
    // op_params[0] = 897988541;s, sizeof(float));
    // memcpy(&epsilon_, op_param)
    normSize_ = normSize;
    weight_.setBackend(bn);
}

ErrorCode CPURMSNorm::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // RMSNorm is similar to LayerNorm which operates on the channel dimension.
    assert(normSize_ == inputs[0]->dimension());
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
#pragma omp parallel for collapse(3) num_threads(thread_count)
    for (int h = 0; h < head; h++) {
        for (int n = 0; n < batch; n++) {
            for (int s = 0; s < seq; s++) {
                double sum_squares = 0.0F;
                // sum
                for (int d = 0; d < dim; d++) {
                    float value = input->dataAt<float>(n, h, s, d);
                    sum_squares += (double)value * value;
                }
                const float mean = sum_squares / dim;
                const float rms = 1.0f / sqrtf(mean + epsilon_);

                memcpy(outputs[0]->ptrAt<float>(n, h, s, 0),
                       inputs[0]->ptrAt<float>(n, h, s, 0),
                       dim * sizeof(float));
                vec_scale_f32(dim, outputs[0]->ptrAt<float>(n, h, s, 0), rms);
            }
        }
    }

#pragma omp parallel for collapse(4) num_threads(thread_count)
    for (int h = 0; h < head; h++) {
        for (int n = 0; n < batch; n++) {
            for (int s = 0; s < seq; s++) {
                for (int d = 0; d < dim; d++) {
                    float weight = weight_.dataAt<float>(0, 0, 0, d);
                    if (add_unit_offset_) {
                        *outputs[0]->ptrAt<float>(n, h, s, d) *= (1 + weight);
                    } else {
                        *outputs[0]->ptrAt<float>(n, h, s, d) *= (weight);
                    }
                }
            }
        }
    }
    return Op::execute(inputs, outputs);
}
ErrorCode CPURMSNorm::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, 1, normSize_); //
    if (loader.getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        // auto l = loader.length(weight_.name());
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