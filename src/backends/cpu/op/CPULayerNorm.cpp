//
// Created by Xiang Li on 2023/11/26.
//

#include "CPULayerNorm.hpp"

namespace mllm {
CPULayerNorm::CPULayerNorm(Backend *bn, string opName, int normSize, bool bias, float epsilon, int threadCount) :
    thread_count(threadCount),
    Op(bn, std::move(opName)), epsilon_(epsilon), bias(bias) {
    normSize_ = normSize;
    weight_.setBackend(bn);
    if (bias) {
        bias_.setBackend(bn);
    }
}
ErrorCode CPULayerNorm::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, 1, normSize_); //
    if (loader.getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        loader.load(&weight_);
    } else {
        weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
    }
    if (bias) {
        bias_.setName(name() + ".bias");
        bias_.reshape(1, 1, 1, normSize_); //
        if (loader.getDataType(bias_.name()) != MLLM_TYPE_COUNT) {
            bias_.setDtype(loader.getDataType(bias_.name()));
            bias_.alloc();
            loader.load(&bias_);
        } else {
            bias_.setDtype(MLLM_TYPE_F32);
            bias_.alloc();
        }
    }

    return Op::load(loader);
}
ErrorCode CPULayerNorm::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(normSize_ == inputs[0]->dimension());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPULayerNorm::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    int batch = input->batch();
    int dim = input->dimension();
    int seq = input->sequence();
    int head = input->head();
    for (int h = 0; h < head; h++) {
        for (int n = 0; n < batch; n++) {
            for (int s = 0; s < seq; s++) {
                float sum_squares = 0.0F;
                float sum = 0.0F;
                // sum
                // #pragma omp parallel for reduction(+ : sum_squares) reduction(+ : sum) num_threads(thread_count)
                for (int d = 0; d < dim; d++) {
                    float value = input->dataAt<float>(n, h, s, d);
                    sum += value;
                }
                float mean = sum / dim;
                // #pragma omp parallel for reduction(+ : sum_squares) num_threads(thread_count)
                for (int d = 0; d < dim; d++) {
                    float value = input->dataAt<float>(n, h, s, d);
                    sum_squares += (value - mean) * (value - mean);
                    output->setDataAt(n, h, s, d, value - mean);
                }
                float rms = std::sqrt(sum_squares / dim + epsilon_);
#pragma omp parallel for num_threads(thread_count)
                for (int d = 0; d < dim; d++) {
                    float value = output->dataAt<float>(n, h, s, d);
                    if (bias) {
                        output->setDataAt<float>(n, h, s, d, weight_.dataAt<float>(0, 0, 0, d) * value / rms + bias_.dataAt<float>(0, 0, 0, d));
                    } else {
                        output->setDataAt<float>(n, h, s, d, weight_.dataAt<float>(0, 0, 0, d) * value / rms);
                    }
                }
            }
        }
    }

    return Op::execute(inputs, outputs);
}
ErrorCode CPULayerNorm::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}
} // namespace mllm