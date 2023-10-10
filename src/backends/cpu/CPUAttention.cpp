//
// Created by ey on 23-9-28.
//

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

CPUAttention::CPUAttention(Backend *bn, int embedding_size, int hidden_size, int head_size, bool multiThread) :
    Op(bn) {
    embedding_size_ = embedding_size;
    hidden_size_ = hidden_size;
    head_size_ = head_size;
    support_multi_thread_ = multiThread;
    Q_proj_.reset(new CPULinear(bn, embedding_size_, hidden_size_ * head_size_, false, false));
    Q_proj_->setName(name() + ".q_proj");
    K_proj_.reset(new CPULinear(bn, embedding_size_, hidden_size_ * head_size_, false, false));
    K_proj_->setName(name() + ".k_proj");
    V_proj_.reset(new CPULinear(bn, embedding_size_, hidden_size_ * head_size_, false, false));
    V_proj_->setName(name() + ".v_proj");
    q_rope_.reset(new CPURoPE(bn, false));
    q_rope_->setName(name() + ".q_rope");
    k_rope_.reset(new CPURoPE(bn, false));
    k_rope_->setName(name() + ".k_rope");
    kq_matmul_.reset(new CPUMatmul(bn, false, true, false));
    kq_matmul_->setName(name() + ".kq_matmul");
    scale_.reset(new CPUScale(bn, 1.0, 0.0, true, false));
    scale_->setName(name() + ".scale");
    softmax_.reset(new CPUSoftMax(bn, 3, false));
    softmax_->setName(name() + ".softmax");
    s_v_matmul_.reset(new CPUMatmul(bn, true, false, false));
    s_v_matmul_->setName(name() + ".s_v_matmul");
    O_proj_.reset(new CPULinear(bn, hidden_size_ * head_size_, embedding_size_, false, false));
    O_proj_->setName(name() + ".O_proj");

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
ErrorCode CPUAttention::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    vector<shared_ptr<Tensor>> q__ = {q_};
    Q_proj_->reshape(inputs, q__);
    vector<shared_ptr<Tensor>> k__ = {k_};
    K_proj_->reshape(inputs, k__);
    vector<shared_ptr<Tensor>> v__ = {v_};
    V_proj_->reshape(inputs, v__);
    vector<shared_ptr<Tensor>> q_pos__ = {q_pos_};
    q_rope_->reshape(q__, q_pos__);
    vector<shared_ptr<Tensor>> k_pos__ = {k_pos_};
    k_rope_->reshape(k__, k_pos__);
    past_key_value_ = k_cached_->allocted();
    mutilHeadReshape(q_pos_, q_state_, head_size_);
    mutilHeadReshape(k_pos_, k_state_, head_size_);
    mutilHeadReshape(v_, v_state_, head_size_);
    if (!past_key_value_) { // 第一次
        // KQ
        vector<shared_ptr<Tensor>> kq_input = {q_state_, k_state_};
        vector<shared_ptr<Tensor>> kq__ = {kq_};
        kq_matmul_->reshape(kq_input, kq__);
        // scale
        vector<shared_ptr<Tensor>> kq_scale__ = {kq_scale_};
        scale_->reshape(kq__, kq_scale__);
        // softmax
        vector<shared_ptr<Tensor>> kq_softmax__ = {kq_softmax_};
        softmax_->reshape(kq_scale__, kq_softmax__);

        // kqv
        vector<shared_ptr<Tensor>> kq_softmax_v_input = {kq_softmax_, v_state_};
        vector<shared_ptr<Tensor>> kq_softmax_v__ = {kq_softmax_v_};
        s_v_matmul_->reshape(kq_softmax_v_input, kq_softmax_v__);

        mutilHeadDeReshape(kq_softmax_v_, kqv_state_, head_size_);
        vector<shared_ptr<Tensor>> O_input = {kqv_state_};
        O_proj_->reshape(O_input, outputs);
    }
    return NO_ERROR;
}
ErrorCode CPUAttention::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    past_key_value_ = k_cached_->allocted();

    vector<shared_ptr<Tensor>> q__ = {q_};
    Q_proj_->setUp(inputs, q__);
    vector<shared_ptr<Tensor>> k__ = {k_};
    K_proj_->setUp(inputs, k__);
    vector<shared_ptr<Tensor>> v__ = {v_};
    V_proj_->setUp(inputs, v__);
    vector<shared_ptr<Tensor>> q_pos__ = {q_pos_};
    q_rope_->setUp(q__, q_pos__);
    vector<shared_ptr<Tensor>> k_pos__ = {k_pos_};
    k_rope_->setUp(k__, k_pos__);
    // v_ = v_ + v_cached_
    q_state_->alloc();
    k_state_->alloc();
    v_state_->alloc();
    if (!past_key_value_) { // 第一次
        vector<shared_ptr<Tensor>> kq_input = {q_state_, k_state_};
        vector<shared_ptr<Tensor>> kq__ = {kq_};
        kq_matmul_->setUp(kq_input, kq__);

        // scale
        vector<shared_ptr<Tensor>> kq_scale__ = {kq_scale_};
        scale_->setUp(kq__, kq_scale__);
        // softmax
        vector<shared_ptr<Tensor>> kq_softmax__ = {kq_softmax_};
        softmax_->setUp(kq_scale__, kq_softmax__);

        vector<shared_ptr<Tensor>> kq_softmax_v_input = {kq_softmax_, v_state_};
        vector<shared_ptr<Tensor>> kq_softmax_v__ = {kq_softmax_v_};
        s_v_matmul_->setUp(kq_softmax_v_input, kq_softmax_v__);

        kqv_state_->alloc();
        vector<shared_ptr<Tensor>> O_input = {kqv_state_};
        O_proj_->setUp(O_input, outputs);
    }
    return NO_ERROR;
}

ErrorCode CPUAttention::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    past_key_value_ = k_cached_->allocted();
    if (past_key_value_) {
        // k_cached
        mergeCacheReshape(k_state_, k_cached_, k_merged_);
        // v_cached
        mergeCacheReshape(v_state_, v_cached_, v_merged_);
        // KQ
        vector<shared_ptr<Tensor>> kq_input = {k_merged_, q_state_};
        vector<shared_ptr<Tensor>> kq__ = {kq_};
        kq_matmul_->reshape(kq_input, kq__);
        kq_matmul_->setUp(kq_input, kq__);
        // scale
        vector<shared_ptr<Tensor>> kq_scale__ = {kq_scale_};
        scale_->reshape(kq__, kq_scale__);
        scale_->setUp(kq__, kq_scale__);
        // softmax
        vector<shared_ptr<Tensor>> kq_softmax__ = {kq_softmax_};
        softmax_->reshape(kq_scale__, kq_softmax__);
        softmax_->setUp(kq_scale__, kq_softmax__);
        // kqv
        vector<shared_ptr<Tensor>> kq_softmax_v_input = {kq_softmax_, v_merged_};
        vector<shared_ptr<Tensor>> kq_softmax_v__ = {kq_softmax_v_};
        s_v_matmul_->reshape(kq_softmax_v_input, kq_softmax_v__);
        s_v_matmul_->setUp(kq_softmax_v_input, kq_softmax_v__);
        // out
        mutilHeadDeReshape(kq_softmax_v_, kqv_state_, head_size_);
        kqv_state_->alloc();
        vector<shared_ptr<Tensor>> O_input__ = {kqv_state_};
        O_proj_->reshape(O_input__, outputs);
        O_proj_->setUp(O_input__, outputs);
    }

    vector<shared_ptr<Tensor>> q__ = {q_};
    Q_proj_->execute(inputs, q__);
    // add KVcache
    vector<shared_ptr<Tensor>> k__ = {k_};
    K_proj_->execute(inputs, k__);
    // add KVcache
    vector<shared_ptr<Tensor>> v__ = {v_};
    V_proj_->execute(inputs, v__);

    vector<shared_ptr<Tensor>> q_pos__ = {q_pos_};
    q_rope_->execute(q__, q_pos__);
    vector<shared_ptr<Tensor>> k_pos__ = {k_pos_};
    k_rope_->execute(k__, k_pos__);

    mutilHeadReshapeExe(q_pos_, q_state_, head_size_);
    mutilHeadReshapeExe(k_pos_, k_state_, head_size_);
    mutilHeadReshapeExe(v_, v_state_, head_size_);

    vector<shared_ptr<Tensor>> kq_input = {q_state_, k_state_};
    if (past_key_value_) {
        mergeCache(k_state_, k_cached_, k_merged_);
        kq_input = {q_state_, k_merged_};
    }
    vector<shared_ptr<Tensor>> kq__ = {kq_};
    kq_matmul_->execute(kq_input, kq__);

    // scale
    vector<shared_ptr<Tensor>> kq_scale__ = {kq_scale_};
    scale_->execute(kq__, kq_scale__);
    // softmax
    vector<shared_ptr<Tensor>> kq_softmax__ = {kq_softmax_};
    softmax_->execute(kq_scale__, kq_softmax__);

    vector<shared_ptr<Tensor>> kq_softmax_v_input = {kq_softmax_, v_state_};
    if (past_key_value_) {
        mergeCache(v_state_, v_cached_, v_merged_);
        kq_softmax_v_input = {kq_softmax_, v_merged_};
    }
    vector<shared_ptr<Tensor>> kq_softmax_v__ = {kq_softmax_v_};
    s_v_matmul_->execute(kq_softmax_v_input, kq_softmax_v__);

    //    vector<shared_ptr<Tensor>> kq_softmax_v_O_input = {kq_softmax_v_};
    mutilHeadDeReshapeExe(kq_softmax_v_, kqv_state_, head_size_);
    vector<shared_ptr<Tensor>> O_input__ = {kqv_state_};
    O_proj_->execute(O_input__, outputs);
    //    outputs[0]->printData<float>();

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
    std::cout<<"["<<outputs[0]->shape(0)<<","<<outputs[0]->shape(1)<<","<<outputs[0]->shape(2)<<","<<outputs[0]->shape(3)<<"]"<<std::endl;
    return NO_ERROR;
}
ErrorCode CPUAttention::load(ParamLoader &loader) {
    return Op::load(loader);
}
} // namespace mllm