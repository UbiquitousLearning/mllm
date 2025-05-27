//
// Created by Daliang Xu on 2024/04/18.
//

#include "CPUQuantize.hpp"
#include "Types.hpp"
#include "backends/cpu/quantize/QuantizeQ8.hpp"

#include <cassert>
#include <cmath>
#include <utility>

namespace mllm {
CPUQuantize::CPUQuantize(Backend *bn, string opName, DataType type, int threadCount) :
    thread_count(threadCount),
    Op(bn, std::move(opName)) {
    assert(type == MLLM_TYPE_I8 || type == MLLM_TYPE_I16);
    activation_dtype_ = type;
    scale_.setBackend(bn);
}

ErrorCode CPUQuantize::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUQuantize::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    int batch = input->batch();
    int head = input->head();
    int seq = input->sequence();
    int dim = input->dimension();

    float quantScale = 0;
    // quantScale = scale_.hostPtr<float>()[0] / 127.0;
    // quantScale = roundf(quantScale * 100000) / 100000;
    switch (activation_dtype_) {
    case MLLM_TYPE_I8:
        quantScale = scale_.hostPtr<float>()[0] / (pow(2, 7) - 1);
        break;
    case MLLM_TYPE_I16:
        quantScale = scale_.hostPtr<float>()[0] / (pow(2, 15) - 1);
        break;
    default:
        return NOT_SUPPORT;
    }
    // quantScale = roundf(quantScale * 100000) / 100000;

    auto src0 = inputs[0];
    auto out0 = outputs[0];

    if (activation_dtype_ == MLLM_TYPE_I8) {
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < head; h++) {
                for (int s = 0; s < seq; s++) {
                    quantize_row_i8(src0->hostPtr<float>() + src0->offset(b, h, s, 0),
                                    out0->hostPtr<int8_t>() + out0->offset(b, h, s, 0),
                                    dim, quantScale);
                }
            }
        }
    } else if (activation_dtype_ == MLLM_TYPE_I16) {
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < head; h++) {
                for (int s = 0; s < seq; s++) {
                    quantize_row_i16(src0->hostPtr<float>() + src0->offset(b, h, s, 0),
                                     out0->hostPtr<int16_t>() + out0->offset(b, h, s, 0),
                                     dim, quantScale);
                }
            }
        }
    } else {
        return NOT_SUPPORT;
    }

    return Op::execute(inputs, outputs);
}

ErrorCode CPUQuantize::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::setUp(inputs, outputs);
}

ErrorCode CPUQuantize::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}

ErrorCode CPUQuantize::load(AbstructLoader &loader) {
    string scaleName = name();

    std::string wordToRemove = "quantize";
    int pos = scaleName.find(wordToRemove);
    if (pos != -1) {
        scaleName.erase(pos, wordToRemove.length());
    }

    scale_.setName(scaleName + "input_scale");
    scale_.reshape(1, 1, 1, 1);
    scale_.setDtype(MLLM_TYPE_F32);
    scale_.alloc();
    loader.load(&scale_);

    return Op::load(loader);
}
} // namespace mllm