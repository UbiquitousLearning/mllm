
#include "QNNLinearINT8Shadow.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cstdint>
#include <memory>
#include "backends/cpu/compute/Matmul.hpp"

namespace mllm {
QNNLinearINT8Shadow::QNNLinearINT8Shadow(Backend *bn, string opName, int in_features, int out_features, bool bias) :
    QNNCommonOp(bn, opName), in_features_(in_features), out_features_(out_features), support_bias_(bias) {
    weight_.setBackend(bn);
    weightScale_.setBackend(bn);
    outputScale_.setBackend(bn);
    inputScale_.setBackend(bn);

    shadowWeight_.setBackend(bn);
    shadowTransposeWeight_.setBackend(bn);

    inputClip_.setBackend(bn);
    outputClip_.setBackend(bn);

    weight_f32_buffer_.setBackend(bn);
    input_f32_buffer_.setBackend(bn);
}

ErrorCode QNNLinearINT8Shadow::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 3);
    assert(outputs.size() == 1);

    outputs[0]->reshape(inputs[2]->batch(), inputs[2]->head(), inputs[2]->sequence(), inputs[2]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNLinearINT8Shadow::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    inputs[0]->setBackend(qnnBackend_);
    inputs[0]->setDtype(MLLM_TYPE_F32);
    inputs[0]->alloc();

    qnnBackend_->pushOutputBuffers(inputs[0]->hostPtr<uint8_t>());

    inputs[1]->setBackend(qnnBackend_);
    inputs[1]->setDtype(MLLM_TYPE_I8);
    inputs[1]->alloc();

    qnnBackend_->pushOutputBuffers(inputs[1]->hostPtr<uint8_t>());

    inputs[2]->setBackend(qnnBackend_);
    inputs[2]->setDtype(MLLM_TYPE_F32);
    inputs[2]->alloc();

    qnnBackend_->pushOutputBuffers(inputs[2]->hostPtr<uint8_t>());

    return MLLM_NO_ERROR;
}

ErrorCode QNNLinearINT8Shadow::load(AbstructLoader &loader) {
    string opName = name();
    std::string wordToRemove = ".shadow";

    int pos = opName.find(wordToRemove);
    if (pos != -1) {
        opName.erase(pos, wordToRemove.length());
    }

    weight_.setName(opName + ".weight");
    weight_.reshape(1, 1, in_features_, out_features_);
    weight_.setDtype(MLLM_TYPE_I8);
    weight_.alloc();
    loader.load(&weight_);

    weightScale_.setName(opName + ".weight.scale");
    weightScale_.reshape(1, 1, 1, 1);
    weightScale_.setDtype(MLLM_TYPE_F32);
    weightScale_.alloc();
    loader.load(&weightScale_);

    outputScale_.setName(opName + ".output_scale");
    outputScale_.reshape(1, 1, 1, 1);
    outputScale_.setDtype(MLLM_TYPE_F32);
    outputScale_.alloc();
    loader.load(&outputScale_);

    inputScale_.setName(opName + ".input_scale");
    inputScale_.reshape(1, 1, 1, 1);
    inputScale_.setDtype(MLLM_TYPE_F32);
    inputScale_.alloc();
    loader.load(&inputScale_);

    inputClip_.setName(opName + ".clip_input");
    inputClip_.reshape(1, 1, 1, 1);
    inputClip_.setDtype(MLLM_TYPE_I8);
    inputClip_.alloc();
    loader.load(&inputClip_);

    outputClip_.setName(opName + ".clip_output");
    outputClip_.reshape(1, 1, 1, 1);
    outputClip_.setDtype(MLLM_TYPE_I8);
    outputClip_.alloc();
    loader.load(&outputClip_);

    shadowWeight_.setName(opName + ".shadow.weight");
    shadowWeight_.reshape(1, 1, in_features_, out_features_);
    shadowWeight_.setDtype(MLLM_TYPE_I8);
    shadowWeight_.alloc();
 
    shadowTransposeWeight_.setName(opName + ".shadow.transpose_weight");
    shadowTransposeWeight_.reshape(1, 1, out_features_, in_features_);
    shadowTransposeWeight_.setDtype(MLLM_TYPE_I8);
    shadowTransposeWeight_.alloc();

    memcpy(shadowWeight_.hostPtr<int8_t>(), weight_.hostPtr<int8_t>(), in_features_ * out_features_);

    for (int i=0; i<out_features_; i++) {
        for (int j=0; j<in_features_; j++) {

            shadowTransposeWeight_.setDataAt<int8_t>(0,0, i,j, shadowWeight_.dataAt<int8_t>(0,0,j,i));

        }
    }

    weight_f32_buffer_.setName(opName + ".shadow.weight_f32_buffer");
    weight_f32_buffer_.reshape(1, 1, 1, in_features_);
    weight_f32_buffer_.setDtype(MLLM_TYPE_F32);
    weight_f32_buffer_.alloc();

    input_f32_buffer_.setName(opName + ".shadow.input_f32_buffer");
    input_f32_buffer_.reshape(1, 1, 1, in_features_);
    input_f32_buffer_.setDtype(MLLM_TYPE_F32);
    input_f32_buffer_.alloc();

    weight_.free();

    return Op::load(loader);
}

ErrorCode QNNLinearINT8Shadow::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}

ErrorCode QNNLinearINT8Shadow::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto opName = name();

    // inputs[0] linear dequant input __fp32
    // inputs[1] linear quant output  int8_t
    // inputs[2] res sum              float

    float input_scale = inputScale_.dataAt<float>(0, 0, 0, 0);
    float weight_scale = weightScale_.dataAt<float>(0, 0, 0, 0);
    float output_scale = outputScale_.dataAt<float>(0, 0, 0, 0) / 127.0f;

    int8_t input_clip = inputClip_.dataAt<int8_t>(0, 0, 0, 0);
    int8_t output_clip = outputClip_.dataAt<int8_t>(0, 0, 0, 0);

    input_scale = input_scale / 127.0;
    input_scale = roundf(input_scale * 100000) / 100000;

    output_scale = roundf(output_scale * 100000) / 100000;

    memcpy(outputs[0]->hostPtr<float>(), inputs[2]->hostPtr<float>(), inputs[2]->batch() * inputs[2]->head() * inputs[2]->sequence() * inputs[2]->dimension() * sizeof(float));

    // input outliers
    if (!input_clip) {
        for (int i = 0; i < inputs[0]->batch(); i++) {
            for (int h = 0; h < inputs[0]->head(); h++) {
                for (int j = 0; j < inputs[0]->sequence(); j++) {
                    for (int k = 0; k < inputs[0]->dimension(); k++) {
                        float round_value = roundf(inputs[0]->dataAt<float>(i, h, j, k) / input_scale);
                        if (round_value > (127.0 * 1.5) || round_value < (-128.0 * 1.5)) {

#if defined(__ARM_NEON)
                            float origin_value = round_value * input_scale * weight_scale;  
                            float clip_value = std::fmax(std::fmin(round_value, 127), -128) * input_scale * weight_scale;  

                            int w_max = shadowWeight_.dimension();  
                            int vector_size = 4;  
                            
#pragma omp parallel for num_threads(4)  
                            for (int w = 0; w <= w_max - vector_size; w += vector_size) {  
                                // Load shadow weights into a NEON vector  
                                int8x8_t weight_vec_int8 = vld1_s8(shadowWeight_.ptrAt<int8_t>(0, 0, k, w));  
                                int16x8_t weight_vec_int16 = vmovl_s8(weight_vec_int8);  

                                // Convert to float  
                                float32x4_t weight_vec = vcvtq_f32_s32(vmovl_s16(vget_low_s16(weight_vec_int16)));  

                                // Compute origin and clip vectors with NEON  
                                float32x4_t origin_vec = vmulq_n_f32(weight_vec, origin_value);  
                                float32x4_t clip_vec = vmulq_n_f32(weight_vec, clip_value);  

                                // Load previous output values  
                                float32x4_t output_vec = vld1q_f32(outputs[0]->ptrAt<float>(i, h, j, w));  

                                // Calculate and store the result  
                                float32x4_t result_vec = vsubq_f32(origin_vec, clip_vec);  
                                result_vec = vaddq_f32(result_vec, output_vec);  

                                vst1q_f32(outputs[0]->ptrAt<float>(i, h, j, w), result_vec);
                            }  

                            // Handle remaining elements, if any  
                            for (int w = (w_max / vector_size) * vector_size; w < w_max; ++w) {  
                                float origin = origin_value * shadowWeight_.dataAt<int8_t>(0, 0, k, w);  
                                float clip = clip_value * shadowWeight_.dataAt<int8_t>(0, 0, k, w);  

                                outputs[0]->setDataAt<float>(i, h, j, w, origin - clip + outputs[0]->dataAt<float>(i, h, j, w));  
                            }  

#else
                            float origin_value = round_value * input_scale * weight_scale;
                            float clip_value = std::fmax(std::fmin(round_value, 127), -128) * input_scale * weight_scale;

#pragma omp parallel for collapse(1) num_threads(4)
                            for (int w = 0; w < shadowWeight_.dimension(); w++) {
                                // if (!(inputs[1]->dataAt<int8_t>(i, h, j, k) <= -128 ||  inputs[1]->dataAt<int8_t>(i, h, j, k) >= 127)) {

                                float origin = origin_value * shadowWeight_.dataAt<int8_t>(0, 0, k, w);

                                float clip =  clip_value * shadowWeight_.dataAt<int8_t>(0, 0, k, w);

                                outputs[0]->setDataAt<float>(i, h, j, w, origin - clip + outputs[0]->dataAt<float>(i, h, j, w));

                                // }
                            }
#endif
                        }
                    }
                }
            }
        }
    }

    // output outliers
    if (!output_clip) {
        for (int i = 0; i < inputs[1]->batch(); i++) {
            for (int h = 0; h < inputs[1]->head(); h++) {
                for (int j = 0; j < inputs[1]->sequence(); j++) {
// #pragma omp parallel for collapse(1) num_threads(4)
                    for (int k = 0; k < inputs[1]->dimension(); k++) {
                        if (inputs[1]->dataAt<int8_t>(i, h, j, k) <= -128 || inputs[1]->dataAt<int8_t>(i, h, j, k) >= 127) {
                            float sum = 0.0f;

#if defined(__ARM_NEON)
                            shadow_vec_dot_fp32_arm(&sum, inputs[0]->ptrAt<float>(i, h, j, 0), shadowTransposeWeight_.ptrAt<int8_t>(0, 0, k, 0), shadowTransposeWeight_.dimension(), input_scale, weight_scale);
#else
                            

                            for (int w = 0; w < shadowTransposeWeight_.dimension(); w++) {
                                sum += roundf(inputs[0]->dataAt<float>(i, h, j, w) / input_scale) * input_scale * (shadowTransposeWeight_.dataAt<int8_t>(0, 0, k, w) * weight_scale);
                            }
#endif
                            outputs[0]->setDataAt<float>(i, h, j, k, inputs[2]->dataAt<float>(i, h, j, k) - (inputs[1]->dataAt<int8_t>(i, h, j, k) * output_scale) + roundf(sum / output_scale) * output_scale);
                        }
                    }
                }
            }
        }
    }

    return MLLM_NO_ERROR;
}

void QNNLinearINT8Shadow::shadow_vec_dot_fp32_arm(float* s, float* x, int8_t* y, int n, float input_scale, float weight_scale) {

    quantize_round_dequantize_row_i8(x, input_f32_buffer_.hostPtr<float>(), n , input_scale);
    dequantize_row_i8(y, weight_f32_buffer_.hostPtr<float>(), n, weight_scale);
    vec_dot_fp32(n, s, input_f32_buffer_.hostPtr<float>(), weight_f32_buffer_.hostPtr<float>());
}

} // namespace mllm
