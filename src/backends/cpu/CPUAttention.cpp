//
// Created by ey on 23-9-28.
//

#include <cmath>
#include "CPUAttention.hpp"

namespace mllm {

void mergeCacheReshape(shared_ptr<Tensor> &A, shared_ptr<Tensor> &B, shared_ptr<Tensor> &C) {
    int a_dim = A->dimension();
    int a_sen = A->sequence();
    int b_dim = B->dimension();
    int b_sen = B->sequence();
    C->reshape(A->batch(), B->head(), a_sen + b_sen, a_dim);
    C->alloc();
}
void mergeCache(shared_ptr<Tensor> &A, shared_ptr<Tensor> &B, shared_ptr<Tensor> &C) {
    // merge a b to c
    int a_dim = A->dimension();
    int a_sen = A->sequence();
    int c_sen = C->sequence();
    for (int h = 0; h < A->head(); ++h) {
        for (int b = 0; b < A->batch(); ++b) {
            for (int d = 0; d < a_dim; ++d) {
                for (int s = 0; s < c_sen; ++s) {
                    float value = 0;
                    if (s < a_sen) {
                        value = A->dataAt<float>(b, h, s, d);
                    } else {
                        value = B->dataAt<float>(b, h, s - a_dim, d);
                    }
                    C->setDataAt<float>(b, h, s, d, value);
                }
            }
        }
    }
}
void mask(shared_ptr<Tensor>& A){
    int batch_size = A->batch();
    int head_num = A->head();
    int sequence = A->sequence();
    int dimension = A->dimension();
    for (int n = 0; n < batch_size; ++n) {
        for (int h = 0; h < head_num; ++h) {
            for (int s = 0; s < sequence; ++s) {
                for (int d = 0; d < dimension; ++d) {
                    double inf =  0;
                    if(d > s) {
                        inf = -std::numeric_limits<double>::infinity();
                    }
                    A->setDataAt<float>({n, h, s, d}, A->dataAt<float>({n,h,s,d})+inf);
                }
            }
        }
    }
}
CPUAttention::CPUAttention(Backend *bn, string opName, int embedding_size, int hidden_size, int head_size, bool multiThread) :
    Op(bn, opName) {
    // embedding_size == hidden_size *head_size !!
    embedding_size_ = embedding_size;
    hidden_size_ = hidden_size;
    head_size_ = head_size;
    support_multi_thread_ = multiThread;
    Q_proj_.reset(new CPULinear(bn, name() + ".wq", embedding_size_, hidden_size_ * head_size_, false, false));
    K_proj_.reset(new CPULinear(bn, name() + ".wk", embedding_size_, hidden_size_ * head_size_, false, false));
    V_proj_.reset(new CPULinear(bn, name() + ".wv", embedding_size_, hidden_size_ * head_size_, false, false));
    q_view_.reset(new CPUView(bn, name() + ".q_view", {-1, head_size_, -1, -1}, {0, 3, 2, 3}, false));
    k_view_.reset(new CPUView(bn, name() + ".k_view", {-1, head_size_, -1, -1}, {0, 3, 2, 3}, false));
    v_view_.reset(new CPUView(bn, name() + ".v_view", {-1, head_size_, -1, -1}, {0, 3, 2, 3}, false));
    q_rope_.reset(new CPURoPE(bn,name() + ".q_rope", false, false));
    k_rope_.reset(new CPURoPE(bn, name() + ".k_rope", false, false));
    kq_matmul_.reset(new CPUMatmul(bn, name() + ".kq_matmul", false, true, false));
    scale_.reset(new CPUScale(bn, name() + ".scale", 1/std::sqrt(hidden_size), 0.0, true, false));
    softmax_.reset(new CPUSoftMax(bn, name() + ".softmax", 3, false));
    s_v_matmul_.reset(new CPUMatmul(bn, name() + ".s_v_matmul", false, false, false));
    s_v_view_.reset(new CPUView(bn, name() + ".s_v_view", {-1, -1, -1, -1}, {0, -1, 2, 1+3}, false));
    O_proj_.reset(new CPULinear(bn, name() + ".wo", hidden_size_ * head_size_, embedding_size_, false, false));

    q_.reset(new Tensor(bn));
    k_.reset(new Tensor(bn));
    v_.reset(new Tensor(bn));
    q_pos_.reset(new Tensor(bn));
    k_pos_.reset(new Tensor(bn));
    q_state_.reset(new Tensor(bn));
    k_state_.reset(new Tensor(bn));
    v_state_.reset(new Tensor(bn));
    kq_.reset(new Tensor(bn));
    kq_scale_.reset(new Tensor(bn));
    kq_softmax_.reset(new Tensor(bn));
    kq_softmax_v_.reset(new Tensor(bn));
    kqv_state_.reset(new Tensor(bn));

    // used for kv cache k+k_cached=k_merged, v+v_cached=v_merged
    k_cached_.reset(new Tensor(bn));
    v_cached_.reset(new Tensor(bn));
    k_merged_.reset(new Tensor(bn));
    v_merged_.reset(new Tensor(bn));
}
ErrorCode CPUAttention::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    past_key_value_ = k_cached_->allocted();

    Q_proj_->reshape(inputs, {q_});
    K_proj_->reshape(inputs, {k_});
    V_proj_->reshape(inputs, {v_});
    q_view_->reshape({q_}, {q_state_});
    k_view_->reshape({k_}, {k_state_});
    v_view_->reshape({v_}, {v_state_});
    q_rope_->reshape({q_state_}, {q_pos_});
    k_rope_->reshape({k_state_}, {k_pos_});
    if (!past_key_value_) { // 第一次
        // kq
        kq_matmul_->reshape({q_pos_, k_pos_}, {kq_});
        // scale
        scale_->reshape({kq_}, {kq_scale_});
        // softmax
        softmax_->reshape({kq_scale_}, {kq_softmax_});
        // kqv
        s_v_matmul_->reshape({kq_softmax_, v_state_}, {kq_softmax_v_});
        // out
        s_v_view_->reshape({kq_softmax_v_}, {kqv_state_});
        O_proj_->reshape({kqv_state_}, outputs);
    }
    return NO_ERROR;
}
ErrorCode CPUAttention::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    past_key_value_ = k_cached_->allocted();

    Q_proj_->setUp(inputs, {q_});
    K_proj_->setUp(inputs, {k_});
    V_proj_->setUp(inputs, {v_});
    q_view_->setUp({q_}, {q_state_});
    k_view_->setUp({k_}, {k_state_});
    v_view_->setUp({v_}, {v_state_});
    q_rope_->setUp({q_state_}, {q_pos_});
    k_rope_->setUp({k_state_}, {k_pos_});
    // v_ = v_ + v_cached_
    if (!past_key_value_) { // 第一次
        // kq
        kq_matmul_->setUp({q_pos_, k_pos_}, {kq_});
        // scale
        scale_->setUp({kq_}, {kq_scale_});
        // softmax
        softmax_->setUp({kq_scale_}, {kq_softmax_});
        // kqv
        s_v_matmul_->setUp({kq_softmax_, v_state_}, {kq_softmax_v_});
        // out
        s_v_view_->setUp({kq_softmax_v_}, {kqv_state_});
        O_proj_->setUp({kqv_state_}, outputs);
    }
    return NO_ERROR;
}

ErrorCode CPUAttention::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    past_key_value_ = k_cached_->allocted();

    if (past_key_value_) {
        // k_cached
        mergeCacheReshape(k_pos_, k_cached_, k_merged_);
        // v_cached
        mergeCacheReshape(v_state_, v_cached_, v_merged_);
        // kq
        kq_matmul_->reshape({q_state_, k_merged_}, {kq_});
        kq_matmul_->setUp({q_state_, k_merged_}, {kq_});
        // scale
        scale_->reshape({kq_}, {kq_scale_});
        scale_->setUp({kq_}, {kq_scale_});
        // softmax
        softmax_->reshape({kq_scale_}, {kq_softmax_});
        softmax_->setUp({kq_scale_}, {kq_softmax_});
        // kqv
        s_v_matmul_->reshape({kq_softmax_, v_merged_}, {kq_softmax_v_});
        s_v_matmul_->setUp({kq_softmax_, v_merged_}, {kq_softmax_v_});
        // out
        s_v_view_->reshape({kq_softmax_v_}, {kqv_state_});
        s_v_view_->setUp({kq_softmax_v_}, {kqv_state_});

//        O_proj_->reshapeOutputs({kqv_state_}, outputs);
    }
    // forward
    // qkv proj
    Q_proj_->execute(inputs, {q_});
    K_proj_->execute(inputs, {k_});
    V_proj_->execute(inputs, {v_});
    q_view_->execute({q_}, {q_state_});
    k_view_->execute({k_}, {k_state_});
    v_view_->execute({v_}, {v_state_});
    // rope
    q_rope_->execute({q_state_}, {q_pos_});
    k_rope_->execute({k_state_}, {k_pos_});
    // k cache
    vector<shared_ptr<Tensor>> kq_input = {q_pos_, k_pos_};
    if (past_key_value_) {
        mergeCache(k_state_, k_cached_, k_merged_);
        kq_input = {q_pos_, k_merged_};
    }
    // kq
    kq_matmul_->execute(kq_input, {kq_});
    // scale
    scale_->execute({kq_}, {kq_scale_});
    //Mask
    if(inputs[0]->sequence()>1){
        mask(kq_scale_);
    }
    // softmax
    softmax_->execute({kq_scale_}, {kq_softmax_});
    // v cache
    vector<shared_ptr<Tensor>> kq_softmax_v_input = {kq_softmax_, v_state_};
    if (past_key_value_) {
        mergeCache(v_state_, v_cached_, v_merged_);
        kq_softmax_v_input = {kq_softmax_, v_merged_};
    }
    // kqv
    s_v_matmul_->execute(kq_softmax_v_input, {kq_softmax_v_});
    // out
    s_v_view_->execute({kq_softmax_v_}, {kqv_state_});
    O_proj_->execute({kqv_state_}, outputs);

    if (!k_cached_->allocted()) { // 第一次
        k_cached_->reshape(k_state_->shape());
        k_cached_->alloc();
        k_cached_->copyFrom(k_state_);
        v_cached_->reshape(v_state_->shape());
        v_cached_->alloc();
        v_cached_->copyFrom(v_state_);
    } else {
        k_cached_->reshape(k_merged_->shape());
        k_cached_->alloc();
        k_cached_->copyFrom(k_merged_);
        v_cached_->reshape(v_merged_->shape());
        v_cached_->alloc();
        v_cached_->copyFrom(v_merged_);
    }
//    std::cout << "[" << outputs[0]->shape(0) << "," << outputs[0]->shape(1) << "," << outputs[0]->shape(2) << "," << outputs[0]->shape(3) << "]" << std::endl;
//    outputs[0]->fullDataTest();
//    outputs[0]->printData<float>();
    return NO_ERROR;
}
ErrorCode CPUAttention::load(ParamLoader &loader) {
    Q_proj_->load(loader);
    K_proj_->load(loader);
    V_proj_->load(loader);
    O_proj_->load(loader);
    return Op::load(loader);
}
ErrorCode CPUAttention::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    Q_proj_->free(inputs, {q_});
    K_proj_->free(inputs, {k_});
    V_proj_->free(inputs, {v_});
    q_view_->free({q_}, {q_state_});
    k_view_->free({k_}, {k_state_});
    v_view_->free({v_}, {v_state_});
    q_rope_->free({q_state_}, {q_pos_});
    k_rope_->free({k_state_}, {k_pos_});
    kq_matmul_->free({q_pos_, k_pos_}, {kq_});
    scale_->free({kq_}, {kq_scale_});
    softmax_->free({kq_scale_}, {kq_softmax_});
    s_v_matmul_->free({kq_softmax_, v_state_}, {kq_softmax_v_});
    s_v_view_->free({kq_softmax_v_}, {kqv_state_});
    O_proj_->free({kqv_state_}, outputs);
    return Op::free(inputs, outputs);
}
} // namespace mllm