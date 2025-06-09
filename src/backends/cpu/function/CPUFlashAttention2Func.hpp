//
// Created by Rongjie Yi on 25-2-16.
//

#ifndef CPUFA2FUNC_HPP
#define CPUFA2FUNC_HPP
#include "CPUBackend.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "../compute/FlashAttention2.hpp"

namespace mllm {
class Tensor;

class CPUFlashAttention2Func : public TensorFunction {
public:
    void reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override {
        auto q_tensor = inputs[0];
        auto k_tensor = inputs[1];
        auto v_tensor = inputs[2];
        auto o_tensor = outputs[0];
        int batch_size = q_tensor->batch();
        int q_head = q_tensor->head();
        int q_sequence = q_tensor->sequence();
        int dimension = q_tensor->dimension();
        o_tensor->reshape(batch_size, q_head, q_sequence, dimension);
        o_tensor->setDtype(inputs[0]->dtype());
        o_tensor->alloc();
    }
    void execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override {
        auto q_tensor = inputs[0];
        auto k_tensor = inputs[1];
        auto v_tensor = inputs[2];
        auto o_tensor = outputs[0];
        bool causal_mask = (bool)args[0];
        int batch_size = q_tensor->batch();
        int q_head = q_tensor->head();
        int q_sequence = q_tensor->sequence();
        int dimension = q_tensor->dimension();
        int k_head = k_tensor->head();
        int k_sequence = k_tensor->sequence();
        int v_head = v_tensor->head();
        int v_sequence = v_tensor->sequence();
        assert(v_head == k_head && v_sequence == k_sequence);
        bool kv_use_fp32 = k_tensor->dtype() == MLLM_TYPE_F32 ? true : false; // x86只支持FP32
        int threads = CPUBackend::cpu_threads;
        if (threads > v_head) {
            threads = v_head; // 线程数不能超过头数
        }
        int32_t br = q_sequence >= 4 ? 4 : 1;
        int32_t bc = q_sequence >= 4 ? 4 : 1;
        constexpr bool high_precision_exp = false;
        // q_tensor->saveData<float>();
        // k_tensor->saveData<float>();
        // v_tensor->saveData<float>();
        // GQA is not ready
        flash_attention_2_forward(
            q_tensor->hostPtr<void>(), k_tensor->hostPtr<void>(), v_tensor->hostPtr<void>(),
            o_tensor->hostPtr<void>(),                             // 输入输出张量
            batch_size, q_head, q_sequence, k_sequence, dimension, // 基本维度
            causal_mask,                                           // 使用因果掩码
            kv_use_fp32,                                           // 使用FP32(x86必须)
            threads,                                               // 使用4线程
            br,                                                    // 查询分块大小64
            bc,                                                    // 键值分块大小128
            q_head,                                                // 查询头数12
            k_head,                                                // 键值头数4
            high_precision_exp                                     // 使用快速指数近似
        );
        // o_tensor->saveData<float>();
    }
};
} // namespace mllm
#endif // CPUFA2FUNC_HPP