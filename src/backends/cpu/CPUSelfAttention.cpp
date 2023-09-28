//
// Created by ey on 23-9-28.
//

#include "CPUSelfAttention.hpp"

namespace mllm {
CPUSelfAttention::CPUSelfAttention(Backend *bn,int embedding_size, int hidden_size, bool multiThread) :
    Op(bn) {
    embedding_size_ = embedding_size;
    hidden_size_ = hidden_size;
    support_multi_thread_ = multiThread;
    Q_proj_.reset(new CPULinear(bn, embedding_size_, hidden_size_, false, false));
    Q_proj_->setName(name()+".Q_proj");
    K_proj_.reset(new CPULinear(bn, embedding_size_, hidden_size_, false, false));
    K_proj_->setName(name()+".K_proj");
    V_proj_.reset(new CPULinear(bn, embedding_size_, hidden_size_, false, false));
    V_proj_->setName(name()+".V_proj");
    O_proj_.reset(new CPULinear(bn, hidden_size_, hidden_size_, false, false));
    O_proj_->setName(name()+".O_proj");
    kq_matmul_.reset(new CPUMatmul(bn, false, false, false));
    kq_matmul_->setName(name()+".kq_matmul");
    softmax_.reset(new CPUSoftMax(bn, 1, false));
    softmax_->setName(name()+".softmax");
    s_v_matmul_.reset(new CPUMatmul(bn, false, false, false));
    s_v_matmul_->setName(name()+".s_v_matmul");

    q_.reset(new Tensor(bn));
    k_.reset(new Tensor(bn));
    v_.reset(new Tensor(bn));
    kq_.reset(new Tensor(bn));
    kq_softmax_.reset(new Tensor(bn));
    kq_softmax_v_.reset(new Tensor(bn));

//    int maxDepth = 5;
//    matmul_.reset(new StrassenMatmul(backend(), false, maxDepth));
}
ErrorCode CPUSelfAttention::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    vector<shared_ptr<Tensor>> q__= {q_};
    Q_proj_->reshape(inputs, q__);
    // add KVcache
    vector<shared_ptr<Tensor>> k__ = {k_};
    K_proj_->reshape(inputs, k__);
    // add KVcache
    vector<shared_ptr<Tensor>> v__ = {v_};
    V_proj_->reshape(inputs, v__);
    //TODO q_permute
    q_->permute(0, 2, 1, 3, false);// TODO 产生新的
    vector<shared_ptr<Tensor>> kq_input  = {k_,q_};
    vector<shared_ptr<Tensor>> kq__ = {kq_};
    kq_matmul_->reshape(kq_input, kq__);

    // TODO add scale
    vector<shared_ptr<Tensor>> kq_softmax__ = {kq_softmax_};
    softmax_->reshape(kq__, kq_softmax__);

    //TODO v_permute
//    v_->permute(0, 2, 1, 3, false);
    vector<shared_ptr<Tensor>> kq_softmax_v_input = {kq_softmax_, v_};
    vector<shared_ptr<Tensor>> kq_softmax_v__ = {kq_softmax_v_};
    s_v_matmul_->reshape(kq_softmax_v_input, kq_softmax_v__);

    vector<shared_ptr<Tensor>> kq_softmax_v_O_input = {kq_softmax_v_};
    O_proj_->reshape(kq_softmax_v_O_input, outputs);
    return NO_ERROR;
}
ErrorCode CPUSelfAttention::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
//    Q_weight_.alloc();
//    Q_bias_.alloc();
//    K_weight_.alloc();
//    K_bias_.alloc();
//    V_weight_.alloc();
//    V_bias_.alloc();
    //    matmul_->encode(inputs, outputs);

    vector<shared_ptr<Tensor>> q__= {q_};
    Q_proj_->setUp(inputs, q__);
    // add KVcache
    vector<shared_ptr<Tensor>> k__ = {k_};
    K_proj_->setUp(inputs, k__);
    // add KVcache
    vector<shared_ptr<Tensor>> v__ = {v_};
    V_proj_->setUp(inputs, v__);

    vector<shared_ptr<Tensor>> kq_input  = {k_, q_};
    vector<shared_ptr<Tensor>> kq__ = {kq_};
    kq_matmul_->setUp(kq_input, kq__);

    //add scale
    vector<shared_ptr<Tensor>> kq_softmax__ = {kq_softmax_};
    softmax_->setUp(kq__, kq_softmax__);

    vector<shared_ptr<Tensor>> kq_softmax_v_input = {kq_softmax_, v_};
    vector<shared_ptr<Tensor>> kq_softmax_v__ = {kq_softmax_v_};
    s_v_matmul_->setUp(kq_softmax_v_input, kq_softmax_v__);

    vector<shared_ptr<Tensor>> kq_softmax_v_O_input = {kq_softmax_v_};
    O_proj_->setUp(kq_softmax_v_O_input, outputs);
    return NO_ERROR;
}
ErrorCode CPUSelfAttention::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    //    matmul_->execute(inputs, outputs);// inputs[0]*Q_weight+Q_bias, q_
    //    matmul_->execute(inputs, outputs);// inputs[0]*K_weight+K_bias. k_
    //    matmul_->execute(inputs, outputs);// inputs[0]*V_weight+V_bias, v_
    //    matmul_->execute(inputs, outputs);// k_*q_, kq_
    //    scale_()//kq_/sqrt(d_k), kq_scale;
    //    softmax_()//softmax(kq_scale), kq_softmax;
    //    matmul_->execute(inputs, outputs);// kq_softmax*v_, kq_softmax_v_
    //    matmul_->execute(inputs, outputs);// kq_softmax_v_*O_weight+O_bias, kq_softmax_v_O_

    vector<shared_ptr<Tensor>> q__= {q_};
    Q_proj_->execute(inputs, q__);
    // add KVcache
    vector<shared_ptr<Tensor>> k__ = {k_};
    K_proj_->execute(inputs, k__);
    // add KVcache
    vector<shared_ptr<Tensor>> v__ = {v_};
    V_proj_->execute(inputs, v__);

    q_->permute(0, 2, 1, 3);
    vector<shared_ptr<Tensor>> kq_input  = {k_, q_};
    vector<shared_ptr<Tensor>> kq__ = {kq_};
    kq_matmul_->execute(kq_input, kq__);

    //add scale
    vector<shared_ptr<Tensor>> kq_softmax__ = {kq_softmax_};
    softmax_->execute(kq__, kq_softmax__);

    v_->permute(0, 2, 1, 3);
    vector<shared_ptr<Tensor>> kq_softmax_v_input = {kq_softmax_, v_};
    vector<shared_ptr<Tensor>> kq_softmax_v__ = {kq_softmax_v_};
    s_v_matmul_->execute(kq_softmax_v_input, kq_softmax_v__);

    vector<shared_ptr<Tensor>> kq_softmax_v_O_input = {kq_softmax_v_};
    O_proj_->execute(kq_softmax_v_O_input, outputs);

    return NO_ERROR;
}
ErrorCode CPUSelfAttention::load(ParamLoader &loader) {
    return Op::load(loader);
}
} // namespace mllm