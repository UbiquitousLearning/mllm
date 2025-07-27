#include "CPUSigmoid.hpp"
// #include <cmath>
#include "Tensor.hpp"
#include "../compute/Sigmoid.hpp"

namespace mllm {

// static void vec_sigmoid_f32(const int n, float *y, const float *x) {
//     for (int i = 0; i < n; ++i) {
//         y[i] = 1.0f / (1.0f + expf(-x[i]));
//     }
// }

CPUSigmoid::CPUSigmoid(Backend *bn, string opName, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    // 构造函数中没有特殊操作
}

ErrorCode CPUSigmoid::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // Sigmoid 是按元素操作的，所以输出张量的形状与输入张量完全相同
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUSigmoid::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto &input = inputs[0];
    auto &output = outputs[0];

#pragma omp parallel for collapse(3) num_threads(thread_count)
    for (int n = 0; n < input->batch(); ++n) {
        for (int h = 0; h < input->head(); ++h) {
            for (int s = 0; s < input->sequence(); ++s) {
                const float *in_ptr = input->ptrAt<float>(n, h, s, 0);
                float *out_ptr = output->ptrAt<float>(n, h, s, 0);
                vec_sigmoid_f32(input->dimension(), out_ptr, in_ptr);
            }
        }
    }

    return Op::execute(inputs, outputs);
}

} // namespace mllm