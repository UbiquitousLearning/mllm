
#include "CPUSparseLinear.hpp"
#include "../compute/MatmulSparse.hpp"

namespace mllm {

CPUSparseLinear::CPUSparseLinear(Backend *bn, string opName, int in_dim, int out_dim, int threadCount) :
    in_dim_(in_dim),
    out_dim_(out_dim),
    thread_count(threadCount),
    Op(bn, std::move(opName)) {
    weight_.setBackend(bn);
}

ErrorCode CPUSparseLinear::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
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

ErrorCode CPUSparseLinear::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto &x = inputs[0];
    auto &o = outputs[0];

    if (x->count() == 0) {
        return Op::execute(inputs, outputs);
    }

    mat_mul_sparse(x.get(), &weight_, o.get(), thread_count);

    return Op::execute(inputs, outputs);
}

ErrorCode CPUSparseLinear::load(AbstructLoader &loader) {
    // This SPARSELINEAR is different from Linear.
    // The weight is equivalent to the transpose of the Linear's weight
    weight_.setName(name() + ".weight_T");
    auto type = loader.getDataType(weight_.name());
    assert(type != MLLM_TYPE_COUNT);
    weight_.setDtype(type);
    weight_.reshape(1, 1, in_dim_, out_dim_);
    weight_.alloc();
    assert(loader.load(&weight_));
    return Op::load(loader);
}

ErrorCode CPUSparseLinear::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    return Op::free(inputs, outputs);
}
} // namespace mllm
