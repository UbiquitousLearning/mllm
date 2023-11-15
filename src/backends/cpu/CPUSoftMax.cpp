
#include "CPUSoftMax.hpp"
#include "CPUBackend.hpp"
#include "compute/VecScalar.hpp"
#include "quantize/Quantize.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
namespace mllm {

// template class CPUSoftMax;
// template class CPUSoftMax;

CPUSoftMax::CPUSoftMax(Backend *bn, string opName, int axis, bool multiThread) :
    Op(bn, opName) {
    axis_ = axis;
}

ErrorCode CPUSoftMax::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout << name() << "  CPUSoftMax  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    outputs[0]->reshape(inputs[0]->shape(0), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3));
    //outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUSoftMax::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout << name() << "  CPUSoftMax()" << std::endl;
    auto &input = inputs[0];
    auto &output = outputs[0];

    float max_ = -INFINITY;
    // 计算最大值
    // for (int n = 0; n < input->shape(0); ++n) {
    //     for (int c = 0; c < input->shape(1); ++c) {
    //         for (int h = 0; h < input->shape(2); ++h) {
    //             for (int w = 0; w < input->shape(3); ++w) {
    //                 std::vector<int> index = {n, c, h, w};
    //                 if (input->dataAt<float>(index) > max_) {
    // max_ = input->dataAt<float>(index);
    //                 }
    //             }
    //         }
    //     }
    // }
    auto *backend = (CPUBackend *)this->backend();
    const auto *table = backend->getSoftMaxTable();
#pragma omp parallel for reduction(max : max_) num_threads(4)
    for (int i = 0; i < input->count(); i++) {
        max_ = std::max(max_, input->dataAtDangerously<float>(i));
    }
    float sum_exp = 0.0;
#pragma omp parallel for reduction(+ : max_) num_threads(4)
    for (int i = 0; i < input->count(); i++) {
        auto value = input->dataAtDangerously<float>(i);
    }
#pragma omp parallel for reduction(+ : sum_exp) num_threads(4)
    for (int i = 0; i < input->count(); i++) {
        auto value = input->dataAtDangerously<float>(i);
        if (value == -INFINITY) {
            output->setDataAtDangerously<float>(i, 0.0F);
        } else {
            mllm_fp16_t fp16_value = MLLM_FP32_TO_FP16(value - max_);
            uint16_t fp_16_ints;
            memcpy(&fp_16_ints, &fp16_value, sizeof(fp_16_ints));
            const float val = MLLM_FP16_TO_FP32(table[fp_16_ints]);
            output->setDataAtDangerously<float>(i, val);
        }
    }
    sum_exp = 1.0 / sum_exp;
    vec_scalar_fp32_(output->ptrAt<float>(0, 0, 0, 0), sum_exp, output->count());

    // for (int n = 0; n < input->shape(0); ++n) {
    //     for (int c = 0; c < input->shape(1); ++c) {
    //         for (int h = 0; h < input->shape(2); ++h) {
    //             #pragma omp parallel for num_threads(4)
    //             for (int w = 0; w < input->shape(3); ++w) {
    //                 std::vector<int> index = {n, c, h, w};
    //                 int num_classes = input->shape(axis_); // 获取类别数量
    //                 // 计算指定类别的 softmax
    //                 float sum_exp = 0.0;
    //                 for (int j = 0; j < num_classes; j++) {
    //                     index[axis_] = j;
    //                     sum_exp += std::exp(input->dataAt<float>(index) - max_);
    //                 }
    //                 // 将 softmax 结果写入输出Tensor
    //                 for (int j = 0; j < num_classes; j++) {
    //                     index[axis_] = j;
    //                     float softmax_value = std::exp(input->dataAt<float>(index)- max_) / sum_exp;
    //                     output->setDataAt<float>(index, softmax_value);
    //                 }
    //             }
    //         }
    //     }
    // }
    return Op::execute(inputs, outputs);
}

} // namespace mllm
