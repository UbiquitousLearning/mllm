
#include "CPUSparseIdLinear.hpp"

#include <utility>
#include "../compute/MatmulSparse.hpp"

namespace mllm {

CPUSparseIdLinear::CPUSparseIdLinear(Backend *bn, string opName, int in_dim, int out_dim, int threadCount) :
    in_dim_(in_dim),
    out_dim_(out_dim),
    thread_count(threadCount),
    Op(bn, std::move(opName)) {
    weight_.setBackend(bn);
}

ErrorCode CPUSparseIdLinear::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);
    auto &x = inputs[0];
    auto &ids = inputs[1];
    auto &o = outputs[0];
    assert(x->dimension() == in_dim_);
    assert(ids->dimension() == out_dim_);
    assert(ids->sequence() == x->sequence());
    assert(ids->head() == x->head());
    assert(ids->batch() == x->batch());
    if (x->count() == 0) {
        o->reshape(0, 0, 0, 0);
        return Op::reshape(inputs, outputs);
    }

    o->reshape(x->batch(), x->head(), x->sequence(), out_dim_);
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUSparseIdLinear::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //    auto start = mllm::mllm_time_us();
    auto &x = inputs[0];
    auto &ids = inputs[1];
    auto &o = outputs[0];

    if (x->count() == 0) {
        return Op::execute(inputs, outputs);
    }

    sparse_mat_mul_id(x.get(), &weight_, ids.get(), o.get(), thread_count);

    //    auto end = mllm::mllm_time_us();
    //    printf("exec time: %ld us\n", end - start);
    return Op::execute(inputs, outputs);
}

ErrorCode CPUSparseIdLinear::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    auto type = loader.getDataType(weight_.name());
    assert(type != MLLM_TYPE_COUNT);
    weight_.setDtype(type);
    weight_.reshape(1, 1, out_dim_, in_dim_);
    weight_.alloc();
    assert(loader.load(&weight_));
    return Op::load(loader);
}

ErrorCode CPUSparseIdLinear::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    return Op::free(inputs, outputs);
}
} // namespace mllm
