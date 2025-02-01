
#include "CPUSoftMax.hpp"
#include <cmath>
#include "Tensor.hpp"
#include "quantize/Quantize.hpp"
#include "../compute/ActivationFunction.hpp"
namespace mllm {

CPUSoftMax::CPUSoftMax(Backend *bn, string opName, int axis, bool do_causal_mask, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    axis_ = axis;
    do_causal_mask_ = do_causal_mask;
    if (axis_ != DIMENSION && !init_table_exp_f16_flag) {
        init_table_exp_f16();
        init_table_exp_f16_flag = true;
    }
}

ErrorCode CPUSoftMax::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << name() << "  CPUSoftMax  reshape" << std::endl;
    // assert(inputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    // outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUSoftMax::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << name() << "  CPUSoftMax()" << std::endl;
    auto &input = inputs[0];
    auto &output = outputs[0];
    int num_classes_in = -1;
    int old_dim = 0;
    if (inputs.size() > 1) {
        num_classes_in = (int)inputs[1]->dataAt<float>(0, 0, 0, 0);
        old_dim = num_classes_in - input->sequence();
    } else {
#ifndef LLAMAFILE_SGEMM
        old_dim = input->dimension() - input->sequence();
#elif defined(USE_QNN) // used for single param softmax in QNN
        old_dim = input->dimension() - input->sequence();
#endif
    }
    memset(output->hostPtr<float>(), 0, output->count() * sizeof(float));
    if (axis_ == DIMENSION) {
        int num_classes = num_classes_in > 0 ? num_classes_in : input->dimension(); // 获取类别数量
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int n = 0; n < input->batch(); ++n) {
            for (int h = 0; h < input->head(); ++h) {
                for (int s = 0; s < input->sequence(); ++s) {
                    int masked_num_classes = num_classes;
                    if (do_causal_mask_ && input->sequence() > 1) {
                        masked_num_classes = s + 1 + old_dim;
                    }
                    float max = -INFINITY;
                    for (int j = 0; j < masked_num_classes; ++j) {
                        max = MAX(max, input->dataAt<float>(n, h, s, j));
                    }
                    float *dp = output->ptrAt<float>(n, h, s, 0);
                    float sum = mllm_vec_soft_max_f32(masked_num_classes, dp, input->ptrAt<float>(n, h, s, 0), max);
                    sum = 1.0 / sum;
                    vec_scale_f32(masked_num_classes, dp, sum);
                }
            }
        }
    } else {
#pragma omp parallel for collapse(4) num_threads(thread_count)
        for (int n = 0; n < input->batch(); ++n) {
            for (int c = 0; c < input->head(); ++c) {
                for (int h = 0; h < input->sequence(); ++h) {
                    for (int w = 0; w < input->dimension(); ++w) {
                        std::vector<int> index = {n, c, h, w};
                        int num_classes = 0;
                        switch (axis_) {
                        case BATCH:
                            num_classes = input->batch();
                            break;
                        case HEAD:
                            num_classes = input->head();
                            break;
                        case SEQUENCE:
                            num_classes = input->sequence();
                            break;
                        case DIMENSION:
                            num_classes = input->dimension();
                            break;
                        }
                        num_classes = num_classes_in > 0 ? num_classes_in : num_classes;
                        float max = -INFINITY;
                        for (int j = 0; j < num_classes; ++j) {
                            index[axis_] = j;
                            max = MAX(max, input->dataAt<float>(index));
                        }
                        vector<float> dp(num_classes);
                        double sum = 0.0;
                        uint16_t scvt;
                        for (int i = 0; i < num_classes; i++) {
                            if (input->dataAt<float>(index) == -INFINITY) {
                                dp[i] = 0.0f;
                            } else {
                                mllm_fp16_t tmp = MLLM_FP32_TO_FP16(input->dataAt<float>(index) - max);
                                memcpy(&scvt, &tmp, sizeof(scvt));
                                const float val = MLLM_FP16_TO_FP32(table_exp_f16[scvt]);
                                sum += (double)val;
                                dp[i] = val;
                            }
                        }
                        // 将 softmax 结果写入输出Tensor
                        for (int i = 0; i < num_classes; i++) {
                            index[axis_] = i;
                            float softmax_value = dp[i] / sum;
                            output->setDataAt<float>(index, softmax_value);
                        }
                        // for (int i = num_classes; i < input->dimension(); i++) {
                        //     output->setDataAt<float>(index, 0);
                        // }
                    }
                }
            }
        }
    }
    return Op::execute(inputs, outputs);
}

} // namespace mllm
