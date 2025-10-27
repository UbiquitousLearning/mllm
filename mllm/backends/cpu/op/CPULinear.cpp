
#include "CPULinear.hpp"
#include "Types.hpp"
#include <cstddef>
#include <iostream>
#include <vector>
#include "../compute/GemmKleidiai.hpp"
#include "backends/cpu/third_party/ggml/QuantizeQ8.hpp"

namespace mllm {

CPULinear::CPULinear(Backend *bn, string opName, int in_features, int out_features, bool bias, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    in_features_ = in_features;
    out_features_ = out_features;
    support_bias_ = bias;
    thread_count = threadCount;
    weight_.setBackend(bn);
    bias_.setBackend(bn);
}

ErrorCode CPULinear::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << name() << "  CPULinear  reshape" << std::endl;
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    if (inputs[0]->count() == 0 && inputs[0]->sequence() != 0) {
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

ErrorCode CPULinear::load(AbstructLoader &loader) {
    // std::cout << name() << "  CPULinear load" << std::endl;
    bool kai_flag = false;
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, out_features_, in_features_);
    if (loader.getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
        if (loader.getDataType(weight_.name()) == MLLM_TYPE_KLEIDIAI_Q4_0) {
#if defined(__aarch64__) || defined(__arm__) || defined(__arm64__)
            kai_thread_count = thread_count;
            kai_flag = true;
            // out_features_:N
            // in_features_:K
#ifndef KAI_FP16_CAL
            size_t packed_b_size = mllm_kleidai_get_packed_b_qsi4_size(out_features_, in_features_);
#else
            size_t packed_b_size = mllm_kleidai_get_packed_b_qsi4_size_to_fp16(out_features_, in_features_);
#endif
            weight_.reshape(1, 1, 1, packed_b_size);
#else
            std::cerr << "KLEIDIAI_Q4_0 is not supported on this platform!" << std::endl;
            exit(-1);
            return NOT_SUPPORT;
#endif
        }
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        loader.load(&weight_);
    } else {
        if (weight_.name().find('v') != std::string::npos && Op::noLoadWeightsDtype() == MLLM_TYPE_Q4_0_4_4) {
            weight_.setDtype(MLLM_TYPE_Q4_0);
        } else {
            weight_.setDtype(Op::noLoadWeightsDtype());
        }
        weight_.alloc();
    }
    if (support_bias_ && !kai_flag) {
        bias_.setName(name() + ".bias");
        bias_.reshape(1, 1, 1, out_features_);
        if (loader.getDataType(bias_.name()) != MLLM_TYPE_COUNT) {
            bias_.setDtype(loader.getDataType(bias_.name()));
            bias_.alloc();
            loader.load(&bias_);
        } else {
            bias_.setDtype(MLLM_TYPE_F32);
            bias_.alloc();
        }
    }
    return Op::load(loader);
}

ErrorCode CPULinear::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //    auto start = mllm::mllm_time_us();
    if (inputs[0]->count() == 0) {
        return Op::execute(inputs, outputs);
    }
    if (inputs[0]->sequence() != outputs[0]->sequence() && outputs[0]->masterTensor() == nullptr) {
        outputs[0]->reshape(outputs[0]->batch(), outputs[0]->head(), inputs[0]->sequence(), outputs[0]->dimension());
        outputs[0]->alloc();
    }
    // TODO: Q8_0 KVCache can not use!!
    if (outputs[0]->dtype() == MLLM_TYPE_Q8_0) {
        auto tmp_out = std::make_shared<Tensor>(outputs[0]->backend());
        // tmp_out->setBackend(outputs[0]->backend());
        auto b = outputs[0]->batch();
        auto h = outputs[0]->head();
        auto d = outputs[0]->dimension();
        auto s = outputs[0]->sequence();
        tmp_out->chls() = outputs[0]->chls();
        tmp_out->setCtype(outputs[0]->ctype());
        tmp_out->reshape(b, h, s, d);
        tmp_out->setDtype(MLLM_TYPE_F32);
        tmp_out->alloc();
        if (weight_.dtype() == MLLM_TYPE_KLEIDIAI_Q4_0) {
#if defined(__aarch64__) || defined(__arm__) || defined(__arm64__)
            kai_thread_count = thread_count;
            // KLEIDIAI_Q4_0 is a packed type, we need to use a special function to handle it
            for (int b = 0; b < inputs[0]->batch(); b++) {
                auto M = inputs[0]->sequence();
                auto N = outputs[0]->dimension();
                auto K = inputs[0]->dimension();
                if (outputs[0]->dtype() == MLLM_TYPE_F16) {
                    mllm_kleidai_gemm_qsi4_to_fp16(outputs[0]->ptrAt<mllm_fp16_t>(b, 0, 0, 0),
                                                   inputs[0]->ptrAt<float>(b, 0, 0, 0),
                                                   (const uint8_t *)weight_.rawHostPtr(), M, N, K);
                } else {
                    mllm_kleidai_gemm_qsi4(outputs[0]->ptrAt<float>(b, 0, 0, 0),
                                           inputs[0]->ptrAt<float>(b, 0, 0, 0),
                                           (const uint8_t *)weight_.rawHostPtr(), M, N, K);
                }
            }
            return MLLM_NO_ERROR;
#else
            std::cerr << "KLEIDIAI_Q4_0 is not supported on this platform!" << std::endl;
            exit(-1);
            return NOT_SUPPORT;
#endif
        }
        mat_mul(inputs[0].get(), &weight_, tmp_out.get(), support_bias_, &bias_, false, true, thread_count);
        if (tmp_out->ctype() == BSHD) {
#pragma omp parallel for collapse(3) num_threads(thread_count)
            for (int b = 0; b < tmp_out->batch(); b++) {
                for (int h = 0; h < tmp_out->head(); h++) {
                    for (int s = 0; s < tmp_out->sequence(); s++) {
                        quantize_row_q8_0(tmp_out->hostPtr<float>() + tmp_out->offset(b, h, s, 0),
                                          (char *)outputs[0]->rawHostPtr()
                                              + outputs[0]->offset(b, h, s, 0) * sizeof(block_q8_0) / QK8_0,
                                          tmp_out->dimension());
                    }
                }
            }
        } else { // BHDS
#pragma omp parallel for collapse(3) num_threads(thread_count)
            for (int b = 0; b < tmp_out->batch(); b++) {
                for (int h = 0; h < tmp_out->head(); h++) {
                    for (int d = 0; d < tmp_out->dimension(); d++) {
                        quantize_row_q8_0(tmp_out->hostPtr<float>() + tmp_out->offset(b, h, 0, d),
                                          (char *)outputs[0]->rawHostPtr()
                                              + outputs[0]->offset(b, h, 0, d) * sizeof(block_q8_0) / QK8_0,
                                          outputs[0]->sequence());
                    }
                }
            }
        }
    } else {
        if (weight_.dtype() == MLLM_TYPE_KLEIDIAI_Q4_0) {
#if defined(__aarch64__) || defined(__arm__) || defined(__arm64__)
            // KLEIDIAI_Q4_0 is a packed type, we need to use a special function to handle it
            if (outputs[0]->ctype() == BHDS) { //&& outputs[0]->masterTensor() != nullptr && outputs[0]->masterTensor()->ctype() == BHDS) {
                for (int b = 0; b < inputs[0]->batch(); b++) {
                    auto M = inputs[0]->sequence();
                    auto N = outputs[0]->dimension(); // out_features_
                    auto K = inputs[0]->dimension();  // in_features_
                    if (outputs[0]->dtype() == MLLM_TYPE_F16) {
                        // auto out_ptr = outputs[0]->ptrAt<mllm_fp16_t>(b, 0, 0, 0);
                        vector<mllm_fp16_t> out_vec(M * N);
                        auto out_ptr = out_vec.data();
                        mllm_kleidai_gemm_qsi4_to_fp16(out_ptr,
                                                       inputs[0]->ptrAt<float>(b, 0, 0, 0),
                                                       (const uint8_t *)weight_.rawHostPtr(), M, N, K);
#pragma omp parallel for num_threads(thread_count)
                        for (int s = 0; s < M; s++) {
                            for (int d = 0; d < N; d++) {
                                outputs[0]->setDataAt<mllm_fp16_t>(b, 0, s, d, out_ptr[s * N + d]);
                            }
                        }
                    } else {
                        // auto out_ptr = outputs[0]->ptrAt<float>(b, 0, 0, 0);
                        vector<float> out_vec(M * N);
                        auto out_ptr = out_vec.data();
                        mllm_kleidai_gemm_qsi4(out_ptr,
                                               inputs[0]->ptrAt<float>(b, 0, 0, 0),
                                               (const uint8_t *)weight_.rawHostPtr(), M, N, K);
#pragma omp parallel for num_threads(thread_count)
                        for (int s = 0; s < M; s++) {
                            for (int d = 0; d < N; d++) {
                                outputs[0]->setDataAt<float>(b, 0, s, d, out_ptr[s * N + d]);
                            }
                        }
                    }
                }
            } else {
                for (int b = 0; b < inputs[0]->batch(); b++) {
                    auto M = inputs[0]->sequence();
                    auto N = outputs[0]->dimension(); // out_features_
                    auto K = inputs[0]->dimension();  // in_features_
                    if (outputs[0]->dtype() == MLLM_TYPE_F16) {
                        mllm_kleidai_gemm_qsi4_to_fp16(outputs[0]->ptrAt<mllm_fp16_t>(b, 0, 0, 0),
                                                       inputs[0]->ptrAt<float>(b, 0, 0, 0),
                                                       (const uint8_t *)weight_.rawHostPtr(), M, N, K);
                    } else {
                        mllm_kleidai_gemm_qsi4(outputs[0]->ptrAt<float>(b, 0, 0, 0),
                                               inputs[0]->ptrAt<float>(b, 0, 0, 0),
                                               (const uint8_t *)weight_.rawHostPtr(), M, N, K);
                    }
                }
            }
            return MLLM_NO_ERROR;
#else
            std::cerr << "KLEIDIAI_Q4_0 is not supported on this platform!" << std::endl;
            exit(-1);
            return NOT_SUPPORT;
#endif
        }
        mat_mul(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_, false, true, thread_count);
    }
    return Op::execute(inputs, outputs);
}
ErrorCode CPULinear::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    for (auto &output : outputs) {
        output->setDtype(activation_dtype_);
        output->alloc();
        // if (weight_.dtype() == MLLM_TYPE_KLEIDIAI_Q4_0 || weight_.dtype() == MLLM_TYPE_Q4_0_4_4) {
        //     output->allowAggregated() = false;
        // }
    }
    return MLLM_NO_ERROR;
}
ErrorCode CPULinear::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.unload();
    if (support_bias_) {
        bias_.unload();
    }
    return Op::free(inputs, outputs);
}

} // namespace mllm
