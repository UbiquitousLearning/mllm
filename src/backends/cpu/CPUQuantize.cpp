//
// Created by Daliang Xu on 2024/04/18.
//

#include "CPUQuantize.hpp"
#include "compute/Matmul.hpp"

#include <utility>

namespace mllm {
CPUQuantize::CPUQuantize(Backend *bn, string opName, int threadCount):thread_count(threadCount), Op(bn, std::move(opName))  {
    activation_dtype_ = MLLM_TYPE_I8;
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
    quantScale = scale_.hostPtr<float>()[0]  / 127.0;
    quantScale = roundf(quantScale * 100000) / 100000;

    auto src0 = inputs[0];
    auto src0_i8 = outputs[0];

// #pragma omp parallel for collapse(4)
    // for (int b = 0; b <batch ; ++b) {
    //     for (int h = 0; h < head; ++h) {
    //         for (int s = 0; s < seq; ++s) {
    //             for (int d = 0; d < dim; ++d) {
    //                 float value = input->dataAt<float>(b, h, s, d);
    //                 int32_t v = static_cast<int32_t>(Round(value / quantScale));
    //                 v = std::max (std::min(v, 127), -128);
    //                 output->setDataAt<uint8_t>(b, h, s, d, static_cast<uint8_t>(v));
    //             }
    //         }
    //         std::cout << std::endl;
    //     }
    // }

#pragma omp parallel for collapse(3) num_threads(thread_count)
    for (int b = 0; b < batch; b++) {
        for (int h = 0; h <head; h++) {
            for (int s = 0; s <seq; s++) {
                quantize_row_i8(src0->hostPtr<float>() + src0->offset(b, h, s, 0),
                                    src0_i8->hostPtr<int8_t>() + src0_i8->offset(b, h, s, 0),
                                    dim, quantScale);
            }
        }
    }
      

    return Op::execute(inputs, outputs);
}

ErrorCode CPUQuantize::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    activation_dtype_ = MLLM_TYPE_I8;
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