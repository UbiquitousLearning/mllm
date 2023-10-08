//
// Created by ey on 23-9-28.
//

#include "CPUSelfAttention.hpp"

namespace mllm {
CPUSelfAttention::CPUSelfAttention(Backend *bn, int embedding_size, int hidden_size, bool multiThread) :
    Op(bn) {
    embedding_size_ = embedding_size;
    hidden_size_ = hidden_size;
    support_multi_thread_ = multiThread;
    Q_proj_.reset(new CPULinear(bn, embedding_size_, hidden_size_, false, false));
    Q_proj_->setName(name() + ".Q_proj");
    K_proj_.reset(new CPULinear(bn, embedding_size_, hidden_size_, false, false));
    K_proj_->setName(name() + ".K_proj");
    V_proj_.reset(new CPULinear(bn, embedding_size_, hidden_size_, false, false));
    V_proj_->setName(name() + ".V_proj");
    kq_matmul_.reset(new CPUMatmul(bn, true, false, false));
    kq_matmul_->setName(name() + ".kq_matmul");
    scale_.reset(new CPUScale(bn, 1.0, 0.0, true, false));
    scale_->setName(name() + ".scale");
    softmax_.reset(new CPUSoftMax(bn, 1, false));
    softmax_->setName(name() + ".softmax");
    s_v_matmul_.reset(new CPUMatmul(bn, false, false, false));
    s_v_matmul_->setName(name() + ".s_v_matmul");
    O_proj_.reset(new CPULinear(bn, hidden_size_, hidden_size_, false, false));
    O_proj_->setName(name() + ".O_proj");

    q_.reset(new Tensor(bn));
    k_.reset(new Tensor(bn));
    v_.reset(new Tensor(bn));
    kq_.reset(new Tensor(bn));
    kq_scale_.reset(new Tensor(bn));
    kq_softmax_.reset(new Tensor(bn));
    kq_softmax_v_.reset(new Tensor(bn));

    // used for kv cache k+k_cached=k_merged, v+v_cached=v_merged
    k_cached_.reset(new Tensor(bn));
    v_cached_.reset(new Tensor(bn));
    k_merged_.reset(new Tensor(bn));
    v_merged_.reset(new Tensor(bn));

    //    int maxDepth = 5;
    //    matmul_.reset(new StrassenMatmul(backend(), false, maxDepth));
}
ErrorCode CPUSelfAttention::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    vector<shared_ptr<Tensor>> q__ = {q_};
    Q_proj_->reshape(inputs, q__);
    // add KVcache
    vector<shared_ptr<Tensor>> k__ = {k_};
    K_proj_->reshape(inputs, k__);
    // k_ = k_ + k_cached_
    //  add KVcache
    vector<shared_ptr<Tensor>> v__ = {v_};
    V_proj_->reshape(inputs, v__);
    // v_ = v_ + v_cached_
//    if (kvcache_) {
//        if(!k_cached_->allocted()){
//            kvcache_ = false;
//        }
//    }
    kvcached_ = k_cached_->allocted();
    if (!kvcached_){
        // KQ
        vector<shared_ptr<Tensor>> kq_input = {k_, q_};
        vector<shared_ptr<Tensor>> kq__ = {kq_};
        kq_matmul_->reshape(kq_input, kq__);
        // scale
        vector<shared_ptr<Tensor>> kq_scale__ = {kq_scale_};
        scale_->reshape(kq__, kq_scale__);
        // softmax
        vector<shared_ptr<Tensor>> kq_softmax__ = {kq_softmax_};
        softmax_->reshape(kq_scale__, kq_softmax__);

        // kqv
        vector<shared_ptr<Tensor>> kq_softmax_v_input = {v_, kq_softmax_};
        vector<shared_ptr<Tensor>> kq_softmax_v__ = {kq_softmax_v_};
        s_v_matmul_->reshape(kq_softmax_v_input, kq_softmax_v__);

        //    vector<shared_ptr<Tensor>> kq_softmax_v_O_input = {kq_softmax_v_};
        O_proj_->reshape(kq_softmax_v__, outputs);
    }
    return NO_ERROR;
}
ErrorCode CPUSelfAttention::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    kvcached_ = k_cached_->allocted();
    //    Q_weight_.alloc();
    //    Q_bias_.alloc();
    //    K_weight_.alloc();
    //    K_bias_.alloc();
    //    V_weight_.alloc();
    //    V_bias_.alloc();
    //    matmul_->encode(inputs, outputs);

    vector<shared_ptr<Tensor>> q__ = {q_};
    Q_proj_->setUp(inputs, q__);
    // add KVcache
    vector<shared_ptr<Tensor>> k__ = {k_};
    K_proj_->setUp(inputs, k__);
    // k_ = k_ + k_cached_
    //  add KVcache
    vector<shared_ptr<Tensor>> v__ = {v_};
    V_proj_->setUp(inputs, v__);
    // v_ = v_ + v_cached_

    if (!kvcached_) {
        vector<shared_ptr<Tensor>> kq_input = {k_, q_};
        vector<shared_ptr<Tensor>> kq__ = {kq_};
        kq_matmul_->setUp(kq_input, kq__);

        // scale
        vector<shared_ptr<Tensor>> kq_scale__ = {kq_scale_};
        scale_->setUp(kq__, kq_scale__);
        // softmax
        vector<shared_ptr<Tensor>> kq_softmax__ = {kq_softmax_};
        softmax_->setUp(kq_scale__, kq_softmax__);

        vector<shared_ptr<Tensor>> kq_softmax_v_input = {v_, kq_softmax_};
        vector<shared_ptr<Tensor>> kq_softmax_v__ = {kq_softmax_v_};
        s_v_matmul_->setUp(kq_softmax_v_input, kq_softmax_v__);

        //    vector<shared_ptr<Tensor>> kq_softmax_v_O_input = {kq_softmax_v_};
        O_proj_->setUp(kq_softmax_v__, outputs);
    }
    return NO_ERROR;
}

void mergeCacheReshape(shared_ptr<Tensor> &a, shared_ptr<Tensor> &b, shared_ptr<Tensor> &c) {
    int a_channel = a->channels();
    int a_senLen = a->seqLen();
    int b_channel = b->channels();
    int b_senLen = b->seqLen();
    c->reshape(a->batch(), a_channel, a_senLen + b_senLen, a->width());
    c->alloc();
}
void mergeCache(shared_ptr<Tensor> &A, shared_ptr<Tensor> &B, shared_ptr<Tensor> &C) {
    // merge a b to c
    int a_hidden = A->channels();
    int a_senLen = A->seqLen();
    int b_senLen = B->seqLen();
    int c_senLen = C->seqLen();
    for (int b = 0; b < A->batch(); ++b) {
        for (int h = 0; h < a_hidden; ++h) {
            for (int s = 0; s < c_senLen; ++s) {
                float value = 0;
                if (s < a_senLen) {
                    value = A->dataAt<float>(b, h, s, 0);
                } else {
                    value = B->dataAt<float>(b, h, s - a_hidden, 0);
                }
                C->setDataAt<float>(b, h, s, 0, value);
            }
        }
    }
}

//void copyTensor(shared_ptr<Tensor>& A, shared_ptr<Tensor>& B){
//    for (int n = 0; n < A->num(); ++n) {
//        for (int c = 0; c < A->channels(); ++c) {
//            for (int h = 0; h < A->height(); ++h) {
//                for (int w = 0; w < A->width(); ++w) {
//                    B->setDataAt<float>(n,c,h,w,A->dataAt<float>(n,c,h,w));
//                }
//            }
//        }
//    }
//}
ErrorCode CPUSelfAttention::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    kvcached_ = k_cached_->allocted();
    if (kvcached_) {
        // k_cached
        mergeCacheReshape(k_, k_cached_, k_merged_);
        // v_cached
        mergeCacheReshape(v_, v_cached_, v_merged_);
        // KQ
        vector<shared_ptr<Tensor>> kq_input = {k_merged_, q_};
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
        vector<shared_ptr<Tensor>> kq_softmax_v_input = {v_merged_, kq_softmax_};
        vector<shared_ptr<Tensor>> kq_softmax_v__ = {kq_softmax_v_};
        s_v_matmul_->reshape(kq_softmax_v_input, kq_softmax_v__);
        s_v_matmul_->setUp(kq_softmax_v_input, kq_softmax_v__);
        // out
        O_proj_->reshape(kq_softmax_v__, outputs);
        O_proj_->setUp(kq_softmax_v__, outputs);
    }

    vector<shared_ptr<Tensor>> q__ = {q_};
    Q_proj_->execute(inputs, q__);
    // add KVcache
    vector<shared_ptr<Tensor>> k__ = {k_};
    K_proj_->execute(inputs, k__);
    // add KVcache
    vector<shared_ptr<Tensor>> v__ = {v_};
    V_proj_->execute(inputs, v__);

    vector<shared_ptr<Tensor>> kq_input = {k_, q_};
    if (kvcached_) {
        mergeCache(k_, k_cached_, k_merged_);
        kq_input = {k_merged_, q_};
    }
    vector<shared_ptr<Tensor>> kq__ = {kq_};
    kq_matmul_->execute(kq_input, kq__);

    // scale
    vector<shared_ptr<Tensor>> kq_scale__ = {kq_scale_};
    scale_->execute(kq__, kq_scale__);
    // softmax
    vector<shared_ptr<Tensor>> kq_softmax__ = {kq_softmax_};
    softmax_->execute(kq_scale__, kq_softmax__);

    vector<shared_ptr<Tensor>> kq_softmax_v_input = {v_, kq_softmax_};
    if (kvcached_) {
        mergeCache(v_, v_cached_, v_merged_);
        kq_softmax_v_input = {v_merged_, kq_softmax_};
    }
    vector<shared_ptr<Tensor>> kq_softmax_v__ = {kq_softmax_v_};
    s_v_matmul_->execute(kq_softmax_v_input, kq_softmax_v__);

    //    vector<shared_ptr<Tensor>> kq_softmax_v_O_input = {kq_softmax_v_};
    O_proj_->execute(kq_softmax_v__, outputs);
    //    outputs[0]->printData<float>();

    if(!k_cached_->allocted()){
        k_cached_->reshape(k_->shape());
        k_cached_->alloc();
//        copyTensor(k_, k_cached_);
        k_cached_->copyFrom(k_);
        v_cached_->reshape(v_->shape());
        v_cached_->alloc();
//        copyTensor(v_, v_cached_);
        v_cached_->copyFrom(v_);
    }else{
        k_cached_->reshape(k_merged_->shape());
        k_cached_->alloc();
//        copyTensor(k_merged_, k_cached_);
        k_cached_->copyFrom(k_merged_);
        v_cached_->reshape(v_merged_->shape());
        v_cached_->alloc();
//        copyTensor(v_merged_, v_cached_);
        v_cached_->copyFrom(v_merged_);
    }

    return NO_ERROR;
}
ErrorCode CPUSelfAttention::load(ParamLoader &loader) {
    return Op::load(loader);
}
} // namespace mllm