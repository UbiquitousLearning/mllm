//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUMATMULFUNC_HPP
#define CPUMATMULFUNC_HPP

#include "../CPUBackend.hpp"
#include "DataType.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "../compute/Matmul.hpp"
#include "../compute/Arithmetic.hpp"
#include <cassert>
#include <vector>
#include <memory>
#include <algorithm> // For std::equal
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include "../compute/GemmKleidiai.hpp"
#include "../compute/GemmFp.hpp"
#include "../AttentionProfiler.hpp"
#include "../AttentionWorkerExecutor.hpp"
#include "CPUAttentionValueLayout.hpp"

namespace mllm {
class Tensor;

class CPUmmFunction : public Op {
private:
    int thread_count = 4;
    bool attention_worker_ = false;
    float output_divisor_ = 1.0F;

    static void tranTensorChl(Tensor &input) {
        transposeAttentionValueChannels(input);
    }

public:
    CPUmmFunction(Backend *bn, string name, int threadCount,
                  bool attention_worker = false,
                  float output_divisor = 1.0F) :
        Op(bn, name), thread_count(threadCount),
        attention_worker_(attention_worker),
        output_divisor_(output_divisor) {
    }

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        if (inputs[0]->ctype() == BHSD) {
            assert(inputs[0]->ctype() == inputs[1]->ctype());
            outputs[0]->setCtype(BHSD);
        } else if (inputs[1]->chls()[SEQUENCE] != 3) {
            tranTensorChl(*inputs[1]);
        }
        if (!inputs[1]->shape().empty() && !inputs[0]->shape().empty()) {
            assert(inputs[0]->dimension() == inputs[1]->sequence());
        }
        outputs[0]->alloc();
        AttentionWorkerExecutor::warmUp(attention_worker_);
        return MLLM_NO_ERROR;
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        if (inputs[0]->ctype() != BHSD && inputs[1]->chls()[SEQUENCE] != 3) {
            tranTensorChl(*inputs[1]);
            assert(inputs[1]->chls()[SEQUENCE] == 3);
        }
        if (inputs[0]->ctype() == BHSD) {
            assert(inputs[0]->ctype() == inputs[1]->ctype());
            outputs[0]->setCtype(BHSD);
        }
        assert(inputs[0]->dimension() == inputs[1]->sequence());
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[1]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        // 遵从原始 reshape 逻辑，在这里 alloc
        // outputs[0]->alloc();
        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        return AttentionWorkerExecutor::run(attention_worker_, [&]() {
            return executeOnAttentionThread(inputs, outputs);
        });
    }

private:
    struct DiagnosticStats {
        double max_abs = 0.0;
        double mean_abs = 0.0;
        std::size_t non_finite = 0;
    };

    static DiagnosticStats diagnosticStats(Tensor &tensor) {
        DiagnosticStats stats;
        double sum_abs = 0.0;
        for (int b = 0; b < tensor.batch(); ++b) {
            for (int h = 0; h < tensor.head(); ++h) {
                for (int s = 0; s < tensor.sequence(); ++s) {
                    for (int d = 0; d < tensor.dimension(); ++d) {
                        double value = 0.0;
                        if (tensor.dtype() == MLLM_TYPE_F32) {
                            value = tensor.dataAt<float>(b, h, s, d);
                        } else if (tensor.dtype() == MLLM_TYPE_F16) {
                            value = static_cast<float>(
                                tensor.dataAt<mllm_fp16_t>(b, h, s, d));
                        } else {
                            continue;
                        }
                        if (!std::isfinite(value)) {
                            ++stats.non_finite;
                            continue;
                        }
                        const double magnitude = std::abs(value);
                        stats.max_abs = std::max(stats.max_abs, magnitude);
                        sum_abs += magnitude;
                    }
                }
            }
        }
        if (tensor.count() > stats.non_finite) {
            stats.mean_abs = sum_abs / static_cast<double>(
                tensor.count() - stats.non_finite);
        }
        return stats;
    }

    void logAttentionInputs(
        const vector<shared_ptr<Tensor>> &inputs,
        const vector<shared_ptr<Tensor>> &outputs) const {
        const char *diagnostics =
            std::getenv("MLLM_QWEN_ATTN_INPUT_DIAGNOSTICS");
        if (diagnostics == nullptr || std::strcmp(diagnostics, "0") == 0
            || inputs[0]->sequence() <= 1) {
            return;
        }
        const auto lhs = diagnosticStats(*inputs[0]);
        const auto rhs = diagnosticStats(*inputs[1]);
        std::cout << "QWEN_ATTN_INPUT"
                  << " op=" << name()
                  << " lhs_shape=" << inputs[0]->batch() << 'x'
                  << inputs[0]->head() << 'x' << inputs[0]->sequence()
                  << 'x' << inputs[0]->dimension()
                  << " lhs_dtype=" << static_cast<int>(inputs[0]->dtype())
                  << " lhs_max_abs=" << lhs.max_abs
                  << " lhs_mean_abs=" << lhs.mean_abs
                  << " lhs_nonfinite=" << lhs.non_finite
                  << " rhs_shape=" << inputs[1]->batch() << 'x'
                  << inputs[1]->head() << 'x' << inputs[1]->sequence()
                  << 'x' << inputs[1]->dimension()
                  << " rhs_dtype=" << static_cast<int>(inputs[1]->dtype())
                  << " rhs_max_abs=" << rhs.max_abs
                  << " rhs_mean_abs=" << rhs.mean_abs
                  << " rhs_nonfinite=" << rhs.non_finite
                  << " out_shape=" << outputs[0]->batch() << 'x'
                  << outputs[0]->head() << 'x' << outputs[0]->sequence()
                  << 'x' << outputs[0]->dimension() << std::endl;
    }

    void logAttentionOutput(const shared_ptr<Tensor> &output) const {
        const char *diagnostics =
            std::getenv("MLLM_QWEN_ATTN_INPUT_DIAGNOSTICS");
        if (diagnostics == nullptr || std::strcmp(diagnostics, "0") == 0
            || output->sequence() <= 1) {
            return;
        }
        const auto stats = diagnosticStats(*output);
        std::cout << "QWEN_ATTN_OUTPUT"
                  << " op=" << name()
                  << " shape=" << output->batch() << 'x' << output->head()
                  << 'x' << output->sequence() << 'x'
                  << output->dimension()
                  << " dtype=" << static_cast<int>(output->dtype())
                  << " max_abs=" << stats.max_abs
                  << " mean_abs=" << stats.mean_abs
                  << " nonfinite=" << stats.non_finite << std::endl;
    }

    void applyOutputDivisor(const shared_ptr<Tensor> &output) const {
        if (output_divisor_ == 1.0F) return;
        mllm_div_fp32(output->hostPtr<float>(), output_divisor_,
                      output->hostPtr<float>(),
                      static_cast<int>(output->count()));
    }

    ErrorCode executeOnAttentionThread(
        vector<shared_ptr<Tensor>> inputs,
        vector<shared_ptr<Tensor>> outputs) {
        logAttentionInputs(inputs, outputs);
        const bool profile_attention = CPUAttentionProfiler::enabled()
            && inputs[0]->sequence() > 1;
        // PhoneLM uses D=160 and chunked K lengths that are multiples of 64.
        // QK has lhs.dimension()==160, while P*V has output.dimension()==160.
        // Tensor names aren't used because traced cache views can rename them.
        const bool profile_dense_pv = profile_attention
            && inputs[0]->dimension() != 160
            && outputs[0]->dimension() == 160;
        ScopedAttentionProfile profile(
            profile_dense_pv ? AttentionProfileStage::DENSE_PV
                             : AttentionProfileStage::QK,
            profile_attention);
        if (inputs[0]->ctype() == BHSD) {
#ifdef ARM
            auto M = inputs[0]->sequence();
            auto N = inputs[1]->dimension();
            auto K = inputs[0]->dimension();
            size_t packed_b_size = mllm_kleidai_get_packed_b_fp32_size(N, K);
            for (int b = 0; b < inputs[0]->batch(); b++) {
                for (int h = 0; h < inputs[0]->head(); h++) {
                    if (inputs[1]->dtype() == MLLM_TYPE_F32) {
                        std::vector<float> packed_b_data(packed_b_size);
                        mllm_kleidai_pack_b_and_bias_fp32(packed_b_data.data(),
                                                          inputs[1]->ptrAt<float>(b, h, 0, 0),
                                                          nullptr, N, K); // Pass nullptr for bias
                        mllm_kleidai_gemm_fp32(outputs[0]->ptrAt<float>(b, h, 0, 0),
                                               inputs[0]->ptrAt<float>(b, h, 0, 0),
                                               packed_b_data.data(),
                                               M, N, K);
                    } else { // inputs[1]->dtype() == MLLM_TYPE_F16
                        std::vector<mllm_fp16_t> packed_b_data(packed_b_size);
                        mllm_kleidai_pack_b_and_bias_fp16(packed_b_data.data(),
                                                          inputs[1]->ptrAt<mllm_fp16_t>(b, h, 0, 0),
                                                          nullptr, N, K); // Pass nullptr for bias
                        mllm_kleidai_gemm_fp16(outputs[0]->ptrAt<float>(b, h, 0, 0),
                                               inputs[0]->ptrAt<float>(b, h, 0, 0),
                                               packed_b_data.data(),
                                               M, N, K);
                    }
                }
            }
            applyOutputDivisor(outputs[0]);
            logAttentionOutput(outputs[0]);
            return MLLM_NO_ERROR;
#else
            auto M = inputs[0]->sequence();
            auto N = inputs[1]->dimension();
            auto K = inputs[0]->dimension();
            memset(outputs[0]->hostPtr<float>(), 0, outputs[0]->cntSize());
            for (int b = 0; b < inputs[0]->batch(); b++) {
                for (int h = 0; h < inputs[0]->head(); h++) {
                    if (inputs[1]->dtype() == MLLM_TYPE_F32) {
                        gemm_fp32(outputs[0]->ptrAt<float>(b, h, 0, 0),
                                  inputs[0]->ptrAt<float>(b, h, 0, 0),
                                  inputs[1]->ptrAt<float>(b, h, 0, 0),
                                  M, N, K);

                    } else { // inputs[1]->dtype() == MLLM_TYPE_F16
                        gemm_fp32_fp16(outputs[0]->ptrAt<float>(b, h, 0, 0),
                                       inputs[0]->ptrAt<float>(b, h, 0, 0),
                                       inputs[1]->ptrAt<mllm_fp16_t>(b, h, 0, 0),
                                       M, N, K);
                    }
                }
            }
            applyOutputDivisor(outputs[0]);
            logAttentionOutput(outputs[0]);
            return MLLM_NO_ERROR;
#endif
        }
        bool isSame = std::equal(inputs[0]->chls().begin(), inputs[0]->chls().end(), inputs[1]->chls().begin());
        assert(inputs[0]->dtype() == MLLM_TYPE_F32);
        mat_mul(inputs[0].get(), inputs[1].get(), outputs[0].get(), false, nullptr, false, isSame, thread_count);
        applyOutputDivisor(outputs[0]);
        logAttentionOutput(outputs[0]);
        return MLLM_NO_ERROR;
    }
};

class CPUmmFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        const auto attention_it = op_param.find("attention_worker");
        const auto divisor_it = op_param.find("output_divisor");
        return new CPUmmFunction(
            bn, name, threadCount,
            attention_it != op_param.end() && attention_it->second != 0.0F,
            divisor_it == op_param.end() ? 1.0F : divisor_it->second);
    }
};

} // namespace mllm
#endif // CPUMATMULFUNC_HPP
