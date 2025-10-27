//
// Created by Rongjie Yi on 25-2-16.
//

#ifndef CPUFA2FUNC_HPP
#define CPUFA2FUNC_HPP

#include "CPUBackend.hpp"
#include "DataType.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "../compute/FlashAttention2.hpp"
#include "../compute/FlashAttention2H.hpp"
#include <algorithm>

namespace mllm {
class Tensor;

class CPUFlashAttention2Func : public Op {
private:
    int thread_count = 4;
    bool causal_mask_;

public:
    CPUFlashAttention2Func(Backend *bn, string name, int threadCount, bool causal_mask) :
        Op(bn, name), thread_count(threadCount), causal_mask_(causal_mask) {
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        auto q_tensor = inputs[0];
        auto o_tensor = outputs[0];

        int batch_size = q_tensor->batch();
        int q_head = q_tensor->head();
        int q_sequence = q_tensor->sequence();
        int dimension = q_tensor->dimension();

        // for BSHD attention start
        if (inputs[0]->ctype() == BHSD && inputs[1]->ctype() == BHSD && inputs[2]->ctype() == BHSD) {
            o_tensor->setCtype(q_tensor->ctype());
        }
        // for BSHD attention end

        o_tensor->reshape(batch_size, q_head, q_sequence, dimension);
        o_tensor->setDtype(inputs[0]->dtype());
        return ErrorCode::MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        auto q_tensor = inputs[0];
        auto k_tensor = inputs[1];
        auto v_tensor = inputs[2];
        auto o_tensor = outputs[0];

        int batch_size = q_tensor->batch();
        int q_head = q_tensor->head();
        int q_sequence = q_tensor->sequence();
        int dimension = q_tensor->dimension();
        int k_head = k_tensor->head();
        int k_sequence = k_tensor->sequence();
        int v_head = v_tensor->head();
        int v_sequence = v_tensor->sequence();

        assert(v_head == k_head && v_sequence == k_sequence);

        bool kv_use_fp32 = (k_tensor->dtype() == MLLM_TYPE_F32); // x86只支持FP32

        int threads = thread_count;
        threads = std::min(threads, v_head);

        int32_t br = q_sequence >= 4 ? 4 : 1;
        int32_t bc = q_sequence >= 4 ? 4 : 1;
        constexpr bool high_precision_exp = true;
        for (int bch = 0; bch < batch_size; ++bch) {
            void *o_ptr = o_tensor->ptrAt<float>(bch, 0, 0, 0);
            void *q_ptr = q_tensor->ptrAt<float>(bch, 0, 0, 0);
            void *k_ptr;
            void *v_ptr;
            if (kv_use_fp32) {
                k_ptr = k_tensor->ptrAt<float>(bch, 0, 0, 0);
                v_ptr = v_tensor->ptrAt<float>(bch, 0, 0, 0);
            } else {
                k_ptr = k_tensor->ptrAt<mllm_fp16_t>(bch, 0, 0, 0);
                v_ptr = v_tensor->ptrAt<mllm_fp16_t>(bch, 0, 0, 0);
            }
            // for BSHD attention start
            if (inputs[0]->ctype() == BHSD && inputs[1]->ctype() == BHSD && inputs[2]->ctype() == BHSD) {
                int km = k_sequence;
                int vm = v_sequence;
                if (k_tensor->masterTensor() != nullptr && v_tensor->masterTensor() != nullptr) {
                    km = k_tensor->masterTensor()->sequence();
                    vm = v_tensor->masterTensor()->sequence();
                }
                flash_attention_2_forward_h(
                    q_ptr, k_ptr, v_ptr, o_ptr,                   // 输入输出张量
                    1, q_head, q_sequence, k_sequence, dimension, // 基本维度
                    causal_mask_,                                 // 使用因果掩码
                    kv_use_fp32,                                  // 使用FP32(x86必须)
                    threads,                                      // 线程数
                    br,                                           // 查询分块大小
                    bc,                                           // 键值分块大小
                    q_head,                                       // 查询头数
                    k_head,                                       // 键值头数
                    high_precision_exp,                           // 使用快速指数近似
                    q_sequence * dimension,
                    km * dimension,
                    vm * dimension);
                // for BSHD attention end
            } else {
                flash_attention_2_forward(
                    q_ptr, k_ptr, v_ptr, o_ptr,                   // 输入输出张量
                    1, q_head, q_sequence, k_sequence, dimension, // 基本维度
                    causal_mask_,                                 // 使用因果掩码
                    kv_use_fp32,                                  // 使用FP32(x86必须)
                    threads,                                      // 线程数
                    br,                                           // 查询分块大小
                    bc,                                           // 键值分块大小
                    q_head,                                       // 查询头数
                    k_head,                                       // 键值头数
                    high_precision_exp                            // 使用快速指数近似
                );
            }
        }
        return ErrorCode::MLLM_NO_ERROR;
    }
};

class CPUFlashAttention2FuncCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        bool causal_mask = (bool)op_param.at("causal_mask");
        return new CPUFlashAttention2Func(bn, name, threadCount, causal_mask);
    }
};

} // namespace mllm
#endif // CPUFA2FUNC_HPP