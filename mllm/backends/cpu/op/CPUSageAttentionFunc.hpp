//
// Created by Rongjie Yi on 25-2-16, with adaptations by Gemini.
//

#ifndef CPUSAGEATTENTIONFUNC_HPP
#define CPUSAGEATTENTIONFUNC_HPP

#include "CPUBackend.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include <algorithm>
#include <vector>
#include "../compute/SageAttention.hpp"
#include "../compute/SageAttentionPT.hpp"
#include "../compute/SageAttentionKVQ8.hpp"

namespace mllm {

class CPUSageAttentionFunc : public Op {
private:
    int thread_count_ = 4;
    bool causal_mask_;

public:
    CPUSageAttentionFunc(Backend *bn, string name, int threadCount, bool causal_mask) :
        Op(bn, name), thread_count_(threadCount), causal_mask_(causal_mask) {
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        auto q_tensor = inputs[0];
        auto o_tensor = outputs[0];

        int batch_size = q_tensor->batch();
        int q_head = q_tensor->head();
        int q_sequence = q_tensor->sequence();
        int dimension = q_tensor->dimension();

        o_tensor->setCtype(q_tensor->ctype());
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

        assert(v_tensor->head() == k_head);
        assert(v_tensor->sequence() == k_sequence);

        bool kv_use_fp32 = (k_tensor->dtype() == MLLM_TYPE_F32);

        int threads = thread_count_;
        threads = std::min(threads, q_head);

        int32_t br = q_sequence >= 4 ? 4 : q_sequence;
        int32_t bc = q_sequence >= 4 ? 4 : q_sequence;
        if (dimension % QK8_0F != 0) {
            if (kv_use_fp32) {
                sage_attn_pt_cpu::sage_attention_forward_cpu_dispatch<float>(
                    q_tensor->hostPtr<float>(),
                    k_tensor->hostPtr<float>(),
                    v_tensor->hostPtr<float>(),
                    o_tensor->hostPtr<float>(),
                    batch_size, q_head, k_head,
                    q_sequence, k_sequence, dimension,
                    causal_mask_,
                    threads,
                    br, bc);
            } else {
                sage_attn_pt_cpu::sage_attention_forward_cpu_dispatch<mllm_fp16_t>(
                    q_tensor->hostPtr<float>(),
                    k_tensor->hostPtr<mllm_fp16_t>(),
                    v_tensor->hostPtr<mllm_fp16_t>(),
                    o_tensor->hostPtr<float>(),
                    batch_size, q_head, k_head,
                    q_sequence, k_sequence, dimension,
                    causal_mask_,
                    threads,
                    br, bc);
            }
            return ErrorCode::MLLM_NO_ERROR;
        }
        if (k_tensor->dtype() == MLLM_TYPE_F32 || k_tensor->dtype() == MLLM_TYPE_F16) {
            if (kv_use_fp32) {
                sage_attn_cpu::sage_attention_forward_cpu_dispatch<float>(
                    q_tensor->hostPtr<float>(),
                    k_tensor->hostPtr<float>(),
                    v_tensor->hostPtr<float>(),
                    nullptr,
                    nullptr,
                    o_tensor->hostPtr<float>(),
                    batch_size, q_head, k_head,
                    q_sequence, k_sequence, dimension,
                    causal_mask_,
                    threads,
                    br, bc, k_sequence);
            } else {
                sage_attn_cpu::sage_attention_forward_cpu_dispatch<mllm_fp16_t>(
                    q_tensor->hostPtr<float>(),
                    k_tensor->hostPtr<mllm_fp16_t>(),
                    v_tensor->hostPtr<mllm_fp16_t>(),
                    nullptr,
                    nullptr,
                    o_tensor->hostPtr<float>(),
                    batch_size, q_head, k_head,
                    q_sequence, k_sequence, dimension,
                    causal_mask_,
                    threads,
                    br, bc, k_sequence);
            }
        } else if (k_tensor->dtype() == MLLM_TYPE_Q8_0F) {
            const float *k_mean_ptr = k_tensor->seqMeans().data();
            const float *v_mean_ptr = v_tensor->seqMeans().data();
            seq_attn_kvq8::sage_attention_forward_cpu_dispatch(
                q_tensor->hostPtr<float>(),
                k_tensor->hostPtr<void>(),
                v_tensor->hostPtr<void>(),
                k_mean_ptr,
                v_mean_ptr,
                o_tensor->hostPtr<float>(),
                batch_size, q_head, k_head, q_sequence, k_sequence, dimension,
                causal_mask_, threads, br, bc);
        } else {
            std::cout << "Unsupported K/V dtype: " << k_tensor->dtype() << std::endl;
            return MLLM_NO_ERROR;
        }

        return ErrorCode::MLLM_NO_ERROR;
    }
};

class CPUSageAttentionFuncCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // 从op_param中获取参数
        bool causal_mask = op_param.find("causal_mask") != op_param.end() ? (bool)op_param.at("causal_mask") : true;
        return new CPUSageAttentionFunc(bn, name, threadCount, causal_mask);
    }
};

} // namespace mllm
#endif // CPUSAGEATTENTIONFUNC_HPP
