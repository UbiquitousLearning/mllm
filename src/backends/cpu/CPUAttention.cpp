//
// Created by ey on 23-9-28.
//

#include <cmath>
#include "CPUAttention.hpp"

namespace mllm {

void mutilHeadReshape(shared_ptr<Tensor> &A, shared_ptr<Tensor> &B, int head_num) {
    B->reshape(A->batch(), head_num, A->sequence(), A->dimension() / head_num);
}
void mutilHeadReshapeExe(shared_ptr<Tensor> &A, shared_ptr<Tensor> &B, int head_num) {
    // 获取 A 的相关维度信息
    int batch = A->batch();
    int sequence = A->sequence();
    int dimension = A->dimension();
    // 计算新的维度信息
    int new_dimension = dimension / head_num;
    // 从 A 复制数据到 B
    for (int n = 0; n < batch; ++n) {
        for (int h = 0; h < head_num; ++h) {
            for (int s = 0; s < sequence; ++s) {
                for (int d = 0; d < new_dimension; ++d) {
                    float value = A->dataAt<float>(n, 0, s, h * new_dimension + d);
                    B->setDataAt<float>(n, h, s, d, value);
                }
            }
        }
    }
}
void mutilHeadDeReshape(shared_ptr<Tensor> &A, shared_ptr<Tensor> &B, int head_num) {
    B->reshape(A->batch(), 1, A->sequence(), A->dimension() * head_num);
}
void mutilHeadDeReshapeExe(shared_ptr<Tensor> &A, shared_ptr<Tensor> &B, int head_num) {
    int batch_size = A->batch();
    int sequence = A->sequence();
    int dimension = A->dimension();

    for (int n = 0; n < batch_size; ++n) {
        for (int s = 0; s < sequence; ++s) {
            for (int d = 0; d < dimension; ++d) {
                for (int h = 0; h < head_num; ++h) {
                    float value = A->dataAt<float>(n, h, s, d);
                    B->setDataAt<float>(n, 0, s, h * dimension + d, value);
                }
            }
        }
    }
}
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
    int b_sen = B->sequence();
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
    q_rope_.reset(new CPURoPE(bn,name() + ".q_rope", false, false));
    k_rope_.reset(new CPURoPE(bn, name() + ".k_rope", false, false));
    kq_matmul_.reset(new CPUMatmul(bn, name() + ".kq_matmul", false, true, false));
    scale_.reset(new CPUScale(bn, name() + ".scale", 1/std::sqrt(hidden_size), 0.0, true, false));
    softmax_.reset(new CPUSoftMax(bn, name() + ".softmax", 3, false));
    s_v_matmul_.reset(new CPUMatmul(bn, name() + ".s_v_matmul", false, false, false));
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
    mutilHeadReshape(q_, q_state_, head_size_);
    mutilHeadReshape(k_, k_state_, head_size_);
    mutilHeadReshape(v_, v_state_, head_size_);
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
        mutilHeadDeReshape(kq_softmax_v_, kqv_state_, head_size_);
        O_proj_->reshape({kqv_state_}, outputs);
    }
    return NO_ERROR;
}
ErrorCode CPUAttention::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    past_key_value_ = k_cached_->allocted();

    Q_proj_->setUp(inputs, {q_});
    K_proj_->setUp(inputs, {k_});
    V_proj_->setUp(inputs, {v_});
    q_state_->alloc();
    k_state_->alloc();
    v_state_->alloc();
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
        kqv_state_->alloc();
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
        kq_matmul_->reshapeOutputs({q_state_, k_merged_}, {kq_});
        // scale
        scale_->reshapeOutputs({kq_}, {kq_scale_});
        // softmax
        softmax_->reshapeOutputs({kq_scale_}, {kq_softmax_});
        // kqv
        s_v_matmul_->reshapeOutputs({kq_softmax_, v_merged_}, {kq_softmax_v_});
        // out
        mutilHeadDeReshape(kq_softmax_v_, kqv_state_, head_size_);
        kqv_state_->alloc();

//        O_proj_->reshapeOutputs({kqv_state_}, outputs);
    }
    // forward
//    inputs[0]->fullData<float>(1);
//    inputs[0]->printData<float>();
    // qkv proj
    Q_proj_->execute(inputs, {q_});
    K_proj_->execute(inputs, {k_});
    V_proj_->execute(inputs, {v_});
    mutilHeadReshapeExe(q_, q_state_, head_size_);
    mutilHeadReshapeExe(k_, k_state_, head_size_);
    mutilHeadReshapeExe(v_, v_state_, head_size_);
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
    mutilHeadDeReshapeExe(kq_softmax_v_, kqv_state_, head_size_);
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
ErrorCode CPUAttention::reshapeOutputs(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout<<name() << "  CPUAttention  reshape" << std::endl;
    Q_proj_->reshapeOutputs(inputs, {q_});
    K_proj_->reshapeOutputs(inputs, {k_});
    V_proj_->reshapeOutputs(inputs, {v_});
    mutilHeadReshape(q_, q_state_, head_size_);
    mutilHeadReshape(k_, k_state_, head_size_);
    mutilHeadReshape(v_, v_state_, head_size_);
    q_state_->alloc();
    k_state_->alloc();
    v_state_->alloc();
    q_rope_->reshapeOutputs({q_state_}, {q_pos_});
    k_rope_->reshapeOutputs({k_state_}, {k_pos_});
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    outputs[0]->alloc();
    return NO_ERROR;
}
ErrorCode CPUAttention::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    Q_proj_->free(inputs, {q_});
    K_proj_->free(inputs, {k_});
    V_proj_->free(inputs, {v_});
    q_state_->free();
    k_state_->free();
    v_state_->free();
    q_rope_->free({q_state_}, {q_pos_});
    k_rope_->free({k_state_}, {k_pos_});
    kq_matmul_->free({q_pos_, k_pos_}, {kq_});
    scale_->free({kq_}, {kq_scale_});
    softmax_->free({kq_scale_}, {kq_softmax_});
    s_v_matmul_->free({kq_softmax_, v_state_}, {kq_softmax_v_});
    kqv_state_->free();
    O_proj_->free({kqv_state_}, outputs);
    return Op::free(inputs, outputs);
}
ErrorCode CPUAttention::setDtype(DataType weight_dtype, DataType activation_dtype) {
    Q_proj_->setDtype(weight_dtype, activation_dtype);
    K_proj_->setDtype(weight_dtype, activation_dtype);
    V_proj_->setDtype(weight_dtype, activation_dtype);
    O_proj_->setDtype(weight_dtype, activation_dtype);
    q_rope_->setDtype(weight_dtype, activation_dtype);
    k_rope_->setDtype(weight_dtype, activation_dtype);
    kq_matmul_->setDtype(weight_dtype, activation_dtype);
    scale_->setDtype(weight_dtype, activation_dtype);
    softmax_->setDtype(weight_dtype, activation_dtype);
    s_v_matmul_->setDtype(weight_dtype, activation_dtype);
    return Op::setDtype(weight_dtype, activation_dtype);
}
} // namespace mllm