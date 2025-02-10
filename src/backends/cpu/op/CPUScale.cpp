
#include "CPUScale.hpp"
#include "../compute/Arithmetic.hpp"

namespace mllm {

CPUScale::CPUScale(Backend *bn, string opName, float scale, float bias, bool bias_after_scale, int threadCount) :
    Op(bn, opName) {
    scale_ = scale;
    bias_ = bias;
    bias_after_scale_ = bias_after_scale;
    thread_count = threadCount;
}

ErrorCode CPUScale::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    // outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUScale::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto &input = inputs[0];
    auto &output = outputs[0];
    if (bias_ == 0.0F) {
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
        for (int n = 0; n < input->batch(); ++n) {
            for (int c = 0; c < input->head(); ++c) {
                for (int h = 0; h < input->sequence(); ++h) {
                    mllm_mul_fp32(input->ptrAt<float>(n, c, h, 0), scale_,
                                  outputs[0]->ptrAt<float>(n, c, h, 0), input->dimension());
                }
            }
        }
        return Op::execute(inputs, outputs);
    }

    if (inputs[0]->masterTensor() == nullptr && outputs[0]->masterTensor() == nullptr && inputs[0]->ctype() == outputs[0]->ctype()) {
        auto copy_size = input->batch() * input->head() * input->sequence() * input->dimension();
        auto in_ptr = inputs[0]->hostPtr<float>();
        auto out_ptr = outputs[0]->hostPtr<float>();
#pragma omp parallel for num_threads(thread_count)
        for (int is = 0; is < copy_size; ++is) {
            if (bias_after_scale_) {
                out_ptr[is] = in_ptr[is] * scale_ + bias_;
            } else {
                out_ptr[is] = (in_ptr[is] + bias_) * scale_;
            }
        }
    } else {
        for (int n = 0; n < input->batch(); ++n) {
            for (int c = 0; c < input->head(); ++c) {
                for (int h = 0; h < input->sequence(); ++h) {
#pragma omp parallel for num_threads(thread_count)
                    for (int w = 0; w < input->dimension(); ++w) {
                        float value = input->dataAt<float>(n, c, h, w);
                        if (bias_after_scale_) {
                            value = value * scale_ + bias_;
                        } else {
                            value = (value + bias_) * scale_;
                        }
                        output->setDataAt<float>(n, c, h, w, value);
                    }
                }
            }
        }
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPUScale::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    // outputs[0]->shallowCopyFrom(inputs[0]);
    if (inputs[0]->masterTensor() == nullptr) {
        inputs[0]->free();
    }
    outputs[0]->setDtype(activation_dtype());
    outputs[0]->alloc();
    inputs[0]->shallowCopyFrom(outputs[0].get(), false);

    return MLLM_NO_ERROR;
}

} // namespace mllm
