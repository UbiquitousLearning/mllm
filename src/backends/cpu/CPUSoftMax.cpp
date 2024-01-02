
#include "CPUSoftMax.hpp"
#include <cmath>
#include "quantize/Quantize.hpp"
#include "compute/VecDot.hpp"
namespace mllm {

//static mllm_fp16_t table_exp_f16[1 << 16];
//static bool init_table_exp_f16_flag = false;
//void init_table_exp_f16() {
//    mllm_fp16_t ii;
//    for (int i = 0; i < (1 << 16); ++i) {
//        uint16_t ui = i;
//        memcpy(&ii, &ui, sizeof(ii));
//        const float f = MLLM_COMPUTE_FP16_TO_FP32(ii);
//        table_exp_f16[i] = MLLM_FP32_TO_FP16(expf(f));
//        //        float val = MLLM_FP16_TO_FP32(expf(f));
//        //        std::cout<<i<<"  "<<f<<" "<<expf(f)<<"  "<<val<<std::endl;
//        //        printf("%d  %f %f  %f\n", i, f, expf(f), val);
//    }
//}

CPUSoftMax::CPUSoftMax(Backend *bn, string opName, int axis, bool multiThread) :
    Op(bn, opName) {
    axis_ = axis;
    if (!init_table_exp_f16_flag) {
        init_table_exp_f16();
        init_table_exp_f16_flag = true;
    }
}

ErrorCode CPUSoftMax::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << name() << "  CPUSoftMax  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    // outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}
inline static void vec_scale_f32(const int n, float *y, const float v) {
    const int np = (n & ~(MLLM_F32_STEP - 1));

    MLLM_F32_VEC vx = MLLM_F32_VEC_SET1(v);

    MLLM_F32_VEC ay[MLLM_F32_ARR];

    for (int i = 0; i < np; i += MLLM_F32_STEP) {
        for (int j = 0; j < MLLM_F32_ARR; j++) {
            ay[j] = MLLM_F32_VEC_LOAD(y + i + j * MLLM_F32_EPR);
            ay[j] = MLLM_F32_VEC_MUL(ay[j], vx);

            MLLM_F32_VEC_STORE(y + i + j * MLLM_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] *= v;
    }

    //    for (int i = 0; i < n; ++i) {
    //        y[i] *= v;
    //    }
}

ErrorCode CPUSoftMax::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << name() << "  CPUSoftMax()" << std::endl;
    auto &input = inputs[0];
    auto &output = outputs[0];

    if (axis_ == DIMENSION) {
        for (int n = 0; n < input->batch(); ++n) {
            #pragma omp parallel for num_threads(4)
            for (int h = 0; h < input->head(); ++h) {
                for (int s = 0; s < input->sequence(); ++s) {
                    int num_classes = input->dimension(); // 获取类别数量
                    float max = -INFINITY;
                    // #pragma omp parallel for num_threads(4)
                    for (int j = 0; j < num_classes; ++j) {
                        max = MAX(max, input->dataAt<float>(n, h, s, j));
                    }
                    float *dp = output->ptrAt<float>(n, h, s, 0);
                    double sum = 0.0;
                    uint16_t scvt;
                    for (int i = 0; i < num_classes; i++) {
                        if (input->dataAt<float>(n, h, s, i) == -INFINITY) {
                            dp[i] = 0.0F;
                        } else {
                            mllm_fp16_t tmp = MLLM_FP32_TO_FP16(input->dataAt<float>(n, h, s, i) - max);
                            memcpy(&scvt, &tmp, sizeof(scvt));
                            const float val = MLLM_FP16_TO_FP32(table_exp_f16[scvt]);
                            sum += (double)val;
                            dp[i] = val;
                        }
                    }

                    sum = 1.0 / sum;
                    vec_scale_f32(num_classes, dp, sum);
                }
            }
        }
    } else {
        for (int n = 0; n < input->batch(); ++n) {
            for (int c = 0; c < input->head(); ++c) {
                for (int h = 0; h < input->sequence(); ++h) {
                    // #pragma omp parallel for num_threads(4)
                    for (int w = 0; w < input->dimension(); ++w) {
                        std::vector<int> index = {n, c, h, w};
                        int num_classes = 0; //input->shape(axis_); // 获取类别数量
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
                    }
                }
            }
        }
    }

    return Op::execute(inputs, outputs);
}

} // namespace mllm
