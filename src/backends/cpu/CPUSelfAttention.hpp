//
// Created by ey on 23-9-28.
//

#ifndef MLLM_CPUSELFATTENTION_HPP
#define MLLM_CPUSELFATTENTION_HPP

#include "Op.hpp"
#include "CPUBackend.hpp"
#include "compute/StrassenMatmul.hpp"
#include "CPULinear.hpp"
#include "CPUMatmul.hpp"
#include "CPUSoftMax.hpp"
#include "CPUScale.hpp"

namespace mllm {

class CPUSelfAttention  final : public Op {
public:
    CPUSelfAttention(Backend *bn, int embedding_size, int hidden_size, bool multiThread);
    virtual ~CPUSelfAttention() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
    
    virtual ErrorCode load(ParamLoader &loader) override;

private:
//    Tensor Q_weight_;
//    Tensor Q_bias_;
//    Tensor K_weight_;
//    Tensor K_bias_;
//    Tensor V_weight_;
//    Tensor V_bias_;
//    std::shared_ptr<StrassenMatmul> matmul_;

    shared_ptr<CPULinear> Q_proj_;
    shared_ptr<CPULinear> K_proj_;
    shared_ptr<CPULinear> V_proj_;
    shared_ptr<CPULinear> O_proj_;
    shared_ptr<CPUMatmul> kq_matmul_;
    shared_ptr<CPUScale> scale_;
    shared_ptr<CPUSoftMax> softmax_;
    shared_ptr<CPUMatmul> s_v_matmul_;

    shared_ptr<Tensor>  q_;
    shared_ptr<Tensor>  k_;
    shared_ptr<Tensor>  v_;
    shared_ptr<Tensor>  kq_;
    shared_ptr<Tensor>  kq_scale_;
    shared_ptr<Tensor>  kq_softmax_;
    shared_ptr<Tensor>  kq_softmax_v_;
//    shared_ptr<Tensor>  kq_softmax_v_O_;

    int embedding_size_;
    int hidden_size_;
    bool support_multi_thread_ = false;
};

class CPUSelfAttentionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn) const {
        int embedding_size = op_param["embedding_size"];
        int hidden_size = op_param["hidden_size"];
        return new CPUSelfAttention(bn, embedding_size, hidden_size, false);
    }
};

} // namespace mllm

#endif // MLLM_CPUSELFATTENTION_HPP
