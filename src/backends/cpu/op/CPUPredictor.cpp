
#include "CPUPredictor.hpp"
#include "../compute/VecDotType.hpp"
#include "../compute/Matmul.hpp"

#include <utility>

namespace mllm {

CPUPredictor::CPUPredictor(Backend *bn, string name, int in_dim, int out_dim, int threadCount) :
    Op(bn, std::move(name)) {
    in_dim_ = in_dim;
    out_dim_ = out_dim;
    r_ = -1; // compute in reshape
    thread_count = threadCount;
    up_.setBackend(bn);
    down_.setBackend(bn);
    hidden_.setBackend(bn);
}

ErrorCode CPUPredictor::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    auto &x = inputs[0];
    auto &o = outputs[0];
    assert(x->dimension() == in_dim_);
    if (x->count() == 0) {
        o->reshape(0, 0, 0, 0);
        return Op::reshape(inputs, outputs);
    }
    o->reshape(x->batch(), x->head(), x->sequence(), out_dim_);
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUPredictor::load(AbstructLoader &loader) {
    up_.setName(name() + ".predictor.up.weight");
    down_.setName(name() + ".predictor.down.weight");
    auto type = loader.getDataType(up_.name());
    assert(type != MLLM_TYPE_COUNT);
    assert(loader.getDataType(down_.name()) == type);
    up_.setDtype(type);
    down_.setDtype(type);

    auto up_size = loader.getTensorSize(up_.name());
    assert(up_size % type_traits[type].size == 0);
    auto n_ele = up_size / type_traits[type].size;
    r_ = (int)(n_ele / in_dim_);
    assert(r_ * out_dim_ * type_traits[type].size == loader.getTensorSize(down_.name()));
    up_.reshape(1, 1, r_, in_dim_);
    down_.reshape(1, 1, out_dim_, r_);
    up_.alloc();
    assert(loader.load(&up_));
    down_.alloc();
    assert(loader.load(&down_));

    return Op::load(loader);
}

ErrorCode CPUPredictor::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto &x = inputs[0];
    auto &o = outputs[0];
    if (x->count() == 0) {
        return Op::execute(inputs, outputs);
    }
    mat_mul(x.get(), &up_, &hidden_, false, nullptr, false, true, thread_count);
    mat_mul(&hidden_, &down_, o.get(), false, nullptr, false, true, thread_count);
    /*
    switch (up_.dtype()) { // TODO: add support to more type
    case MLLM_TYPE_F32: {
        mat_mul_fp32(x.get(), &up_, &hidden_, false, nullptr, false, true, thread_count);
        mat_mul_fp32(&hidden_, &down_, o.get(), false, nullptr, false, true, thread_count);
        break;
    }
    case MLLM_TYPE_F16: {
        mat_mul_fp32_fp16(x.get(), &up_, &hidden_, false, nullptr, false, true, thread_count);
        mat_mul_fp32_fp16(&hidden_, &down_, o.get(), false, nullptr, false, true, thread_count);
        break;
    }
    default:
        fprintf(stderr, "type not support yet");
        assert(false);
    }
    */
    return Op::execute(inputs, outputs);
}

ErrorCode CPUPredictor::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    up_.free();
    down_.free();
    hidden_.free();
    return Op::free(inputs, outputs);
}

ErrorCode CPUPredictor::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    auto &x = inputs[0];
    hidden_.setDtype(MLLM_TYPE_F32);
    hidden_.reshape(x->batch(), x->head(), x->sequence(), r_);
    hidden_.alloc();

    return Op::setUp(inputs, outputs);
}
} // namespace mllm
