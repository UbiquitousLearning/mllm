
#include "CPUNorm.hpp"
#include <cmath>

namespace mllm {

CPUNorm::CPUNorm(Backend *bn, string opName,int L_n, int threadCount) : thread_count(threadCount),
    Op(bn, opName) {
    assert(L_n ==1 || L_n ==2);
    L_n_ = L_n;
}

ErrorCode CPUNorm::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUNorm::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // Get the data from the tensor
    auto data = inputs[0]->hostPtr<float>();

    // Calculate the size of the data
    auto input = inputs[0];
    auto output = outputs[0];
    int batch = input->batch();
    int dim = input->dimension();
    int seq = input->sequence();
    int head = input->head();
#pragma omp parallel for collapse(3) num_threads(thread_count)
    for (int h = 0; h < head; h++) {
        for (int n = 0; n < batch; n++) {
            for (int s = 0; s < seq; s++) {
                if (L_n_ == 2) {
                    // Calculate the sum of squares
                    float sum_of_squares = 0.0f;
// #pragma omp parallel for num_threads(thread_count)
                    for (int d = 0; d < inputs[0]->dimension(); ++d) {
                        sum_of_squares += inputs[0]->dataAt<float>(n, h, s,d) * inputs[0]->dataAt<float>(n, h, s,d);
                    }
                    // Calculate the L2 norm
                    float l2_norm = std::sqrt(sum_of_squares);

                    // Use the L2 norm in your code...
#pragma omp parallel for num_threads(thread_count)
                    for (int d = 0; d < dim; d++) {
                        outputs[0]->setDataAt<float>(n, h, s,d, l2_norm);
                    }
                } else {
                    float sum_of_abs_values = 0.0f;

// #pragma omp parallel for num_threads(thread_count)
                    for (int d = 0; d < inputs[0]->dimension(); ++d) {
                        sum_of_abs_values += std::abs(inputs[0]->dataAt<float>(n, h, s,d));
                    }
#pragma omp parallel for num_threads(thread_count)
                    for (int d = 0; d < dim; d++) {
                        outputs[0]->setDataAt<float>(n, h, s,d, sum_of_abs_values);
                    }

                }
            }
        }
    }
//     int size = inputs[0]->batch() * inputs[0]->head() * inputs[0]->sequence() * inputs[0]->dimension();
//     if (L_n_ == 2) {
//         // Calculate the sum of squares
//         float sum_of_squares = 0.0f;
//
// #pragma omp parallel for num_threads(thread_count)
//         for (int i = 0; i < size; ++i) {
//             sum_of_squares += data[i] * data[i];
//         }
//
//         // Calculate the L2 norm
//         float l2_norm = std::sqrt(sum_of_squares);
//
//         // Use the L2 norm in your code...
//         outputs[0]->setDataAt(0, 0, 0, 0, l2_norm);
//     } else {
//         // Calculate the L1 norm
//         float sum_of_abs_values = 0.0f;
//
// #pragma omp parallel for num_threads(thread_count)
//         for (int i = 0; i < size; ++i) {
//             sum_of_abs_values += std::abs(data[i]);
//         }
//
//         // Use the L1 norm in your code...
//         outputs[0]->setDataAt(0, 0, 0, 0, sum_of_abs_values);
//     }

    return Op::execute(inputs, outputs);
}
} // namespace mllm
