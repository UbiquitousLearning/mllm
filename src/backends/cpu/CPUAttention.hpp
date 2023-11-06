//
// Created by ey on 23-9-28.
//

#ifndef MLLM_CPUATTENTION_HPP
#define MLLM_CPUATTENTION_HPP

#include "Op.hpp"
#include "CPUBackend.hpp"
#include "compute/StrassenMatmul.hpp"
#include "CPULinear.hpp"
#include "CPUMatmul.hpp"
#include "CPUSoftMax.hpp"
#include "CPUScale.hpp"
#include "CPURoPE.hpp"

namespace mllm {

class CPUAttention final : public Op {
public:
    CPUAttention(Backend *bn, string opName, int embedding_size, int hidden_size, int head_size, bool multiThread);
    virtual ~CPUAttention() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode reshapeOutputs(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    //virtual ErrorCode setDtype(DataType activation_dtype) override;

    virtual ErrorCode load(ParamLoader &loader) override;

private:
    shared_ptr<CPULinear> Q_proj_;
    shared_ptr<CPULinear> K_proj_;
    shared_ptr<CPULinear> V_proj_;
    shared_ptr<CPURoPE> q_rope_;
    shared_ptr<CPURoPE> k_rope_;
    shared_ptr<CPUMatmul> kq_matmul_;
    shared_ptr<CPUScale> scale_;
    shared_ptr<CPUSoftMax> softmax_;
    shared_ptr<CPUMatmul> s_v_matmul_;
    shared_ptr<CPULinear> O_proj_;

    shared_ptr<Tensor>  q_;
    shared_ptr<Tensor>  k_;
    shared_ptr<Tensor>  v_;
    shared_ptr<Tensor>  q_pos_;
    shared_ptr<Tensor>  k_pos_;
    shared_ptr<Tensor>  q_state_;
    shared_ptr<Tensor>  k_state_;
    shared_ptr<Tensor>  v_state_;
    shared_ptr<Tensor>  kq_;
    shared_ptr<Tensor>  kq_scale_;
    shared_ptr<Tensor>  kq_softmax_;
    shared_ptr<Tensor>  kq_softmax_v_;
    shared_ptr<Tensor>  kqv_state_;

    shared_ptr<Tensor>  k_cached_;
    shared_ptr<Tensor>  v_cached_;

    shared_ptr<Tensor>  k_merged_;
    shared_ptr<Tensor>  v_merged_;

    bool past_key_value_ = true;

    int embedding_size_;
    int hidden_size_;
    int head_size_;
    bool support_multi_thread_ = false;
};

class CPUAttentionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        int embedding_size = op_param["embedding_size"];
        int hidden_size = op_param["hidden_size"];
        int head_size = op_param["head_size"];
        return new CPUAttention(bn, name, embedding_size, hidden_size, head_size, false);
    }
};

} // namespace mllm

#endif // MLLM_CPUATTENTION_HPP
