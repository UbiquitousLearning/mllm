#include "CPULinearInt8.hpp"
#include "Types.hpp"
#include "../compute/Matmul.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>

namespace mllm {

CPULinearInt8::CPULinearInt8(Backend *bn, string opName, int in_features, int out_features, bool bias, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    in_features_ = in_features;
    out_features_ = out_features;
    support_bias_ = bias;
    thread_count = threadCount;
    weight_.setBackend(bn);
    originWeight_.setBackend(bn);
    bias_.setBackend(bn);

    weightScale_.setBackend(bn);
    biasScale_.setBackend(bn);
    inputActivatationScale_.setBackend(bn);
    outputActivatationScale_.setBackend(bn);
}

ErrorCode CPULinearInt8::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << name() << "  CPULinearInt8  reshape" << std::endl;
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    if (inputs[0]->count() == 0) {
        outputs[0]->reshape(0, 0, 0, 0);
        return Op::reshape(inputs, outputs);
    }
    // N     |    C       |   H                   |  W
    // -----------------------------------------------
    // 1     |out_channel | in_channel            |  1
    //       |out_features| in_features           |
    // -----------------------------------------------
    // batch |in_channel  | seq_len               |  1
    //       |in_features | inputs[0]->sequence()   |
    // -----------------------------------------------
    // batch |out_channel | seq_len               |  1
    //       |out_features|  inputs[0]->sequence()  |
    assert(inputs[0]->head() == 1);
    assert(in_features_ == inputs[0]->dimension());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), out_features_);
    // outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPULinearInt8::load(AbstructLoader &loader) {
    // std::cout << name() << "  CPULinearInt8 load" << std::endl;
    originWeight_.setName(name() + ".weight");
    // origin weight is [in, out], while the linear weight is [out, in]
    originWeight_.reshape(1, 1, in_features_, out_features_);
    if (loader.getDataType(originWeight_.name()) != MLLM_TYPE_COUNT) {
        originWeight_.setDtype(loader.getDataType(originWeight_.name()));
        originWeight_.alloc();
        loader.load(&originWeight_);

        weight_.setName(name() + ".linear.weight");
        weight_.reshape(1, 1, out_features_, in_features_);
        weight_.setDtype(MLLM_TYPE_I8);
        weight_.alloc();

        for (int i = 0; i < in_features_; ++i) {
            for (int j = 0; j < out_features_; ++j) {
                weight_.setDataAt<int8_t>(0, 0, j, i, originWeight_.dataAt<int8_t>(0, 0, i, j));
            }
        }

        originWeight_.free();

        weightScale_.setName(name() + ".weight.scale");
        weightScale_.reshape(1, 1, 1, 1);
        weightScale_.setDtype(MLLM_TYPE_F32);
        weightScale_.alloc();
        loader.load(&weightScale_);

    } else {
        weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
    }
    if (support_bias_) {
        bias_.setName(name() + ".bias");
        bias_.reshape(1, 1, 1, out_features_);
        if (loader.getDataType(bias_.name()) != MLLM_TYPE_COUNT) {
            bias_.setDtype(loader.getDataType(bias_.name()));
            bias_.alloc();
            loader.load(&bias_);

            biasScale_.setName(name() + ".bias.scale");
            biasScale_.reshape(1, 1, 1, 1);
            biasScale_.setDtype(MLLM_TYPE_F32);
            biasScale_.alloc();
            loader.load(&biasScale_);
        } else {
            bias_.setDtype(MLLM_TYPE_F32);
            bias_.alloc();
        }
    }

    inputActivatationScale_.setName(name() + ".input_scale");
    inputActivatationScale_.reshape(1, 1, 1, 1);
    inputActivatationScale_.setDtype(MLLM_TYPE_F32);
    inputActivatationScale_.alloc();
    loader.load(&inputActivatationScale_);

    outputActivatationScale_.setName(name() + ".output_scale");
    outputActivatationScale_.reshape(1, 1, 1, 1);
    outputActivatationScale_.setDtype(MLLM_TYPE_F32);
    outputActivatationScale_.alloc();
    loader.load(&outputActivatationScale_);

    return Op::load(loader);
}

ErrorCode CPULinearInt8::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (inputs[0]->count() == 0) {
        return Op::execute(inputs, outputs);
    }

    assert(weight_.dtype() == MLLM_TYPE_I8);

    mat_mul_fp32_i8(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_, thread_count);

    return Op::execute(inputs, outputs);
}
ErrorCode CPULinearInt8::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    if (support_bias_) {
        bias_.free();
    }
    return Op::free(inputs, outputs);
}

ErrorCode CPULinearInt8::mat_mul_fp32_i8(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, int thread_count) {
    // todo: load scale from loader
    float scale1 = inputActivatationScale_.hostPtr<float>()[0] / 127.0;
    // scale1 = roundf(scale1 * 100000) / 100000;

    float scale2 = weightScale_.hostPtr<float>()[0];

    float scale3 = 0.0;
    if (support_bias_)
        scale3 = biasScale_.hostPtr<float>()[0];

    float scale4 = outputActivatationScale_.hostPtr<float>()[0] / 127.0;
    // scale4 = roundf(scale4 * 100000) / 100000;

    assert(src1->dtype() == MLLM_TYPE_I8);
    assert(src0_->dtype() == MLLM_TYPE_F32);
    Tensor src0_i8(src0_->shape());
    src0_i8.setBackend(src0_->backend());
    src0_i8.setDtype(MLLM_TYPE_I8);
    src0_i8.alloc();

    // as we use the code from q80, the dimension still needs to be multiple of 32
    if (src0_->dimension() % QK8_0 == 0) {
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int b = 0; b < src0_->batch(); b++) {
            for (int h = 0; h < src0_->head(); h++) {
                for (int s = 0; s < src0_->sequence(); s++) {
                    quantize_row_i8(src0_->hostPtr<float>() + src0_->offset(b, h, s, 0),
                                    src0_i8.hostPtr<int8_t>() + src0_i8.offset(b, h, s, 0),
                                    src0_->dimension(), scale1);
                }
            }
        }
    } else {
        std::cout << "[ERROR]: " << src0_->dimension() << "%" << QK8_0 << "!=0" << std::endl;
        assert(src0_->dimension() % QK8_0 == 0);
    }
    auto *src0 = &src0_i8;
    assert(src0->dtype() == MLLM_TYPE_I8);

    const int M = src0->sequence();
    const int K = src0->dimension();
    const int N = src1->sequence();
    Tensor *src0_cal = src0;
    Tensor *src1_cal = src1;
    const int64_t blck_0 = 16;
    // #pragma omp parallel for collapse(4) num_threads(thread_count)
    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
            const int b_1 = (src1->batch() == 1 && src1->head() == 1) ? 0 : b;
            const int h_1 = (src1->batch() == 1 && src1->head() == 1) ? 0 : h;
            for (int m = 0; m < M; m++) {
                const int num_blocks = N / blck_0;
                const int remainder = N % blck_0;
                for (int block = 0; block < num_blocks + 1; block++) {
                    for (int n = block * blck_0; n < (block + 1) * blck_0 & n < num_blocks * blck_0 + remainder; n++) {
                        int s_1, d_1;
                        int s_0, d_0;
                        s_1 = n;
                        d_1 = 0;
                        s_0 = m;
                        d_0 = 0;

                        vec_dot_i8_i8(K, dst->ptrAt<float>(b, h, m, n), src1_cal->hostPtr<int8_t>() + src1_cal->offset(b_1, h_1, s_1, d_1), src0_cal->hostPtr<int8_t>() + src0_cal->offset(b, h, s_0, d_0), scale1, scale2);
                        if (support_bias) {
                            *dst->ptrAt<float>(b, h, m, n) += bias->dataAt<int8_t>(0, 0, 0, n) * scale3;
                        }
                        *dst->ptrAt<float>(b, h, m, n) = std::fmaxf(std::fminf(roundf(*dst->ptrAt<float>(b, h, m, n) / scale4), 127), -128) * scale4;
                    }
                }
            }
        }
    }
    return MLLM_NO_ERROR;
}

} // namespace mllm
