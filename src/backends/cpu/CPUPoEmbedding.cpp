#include "CPUPoEmbedding.hpp"
#include <cassert>

namespace mllm {

CPUPoEmbedding::CPUPoEmbedding(Backend *bn, string opName, int max_num, int hidden_dim, int threadCount) :
    max_num_(max_num), hidden_dim_(hidden_dim), thread_count(threadCount),
    Op(bn, opName) {
}

ErrorCode CPUPoEmbedding::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    assert(inputs[0]->batch() == 1);
    assert(inputs[0]->head() == 1);
    assert(inputs[0]->sequence() <= max_num_);
    assert(inputs[0]->dimension() == hidden_dim_);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUPoEmbedding::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    int H = inputs[0]->sequence();
    int W = inputs[0]->dimension();
    for (int h = 0; h < H; ++h) {
#pragma omp parallel for num_threads(thread_count)
        for (int w = 0; w < W; ++w) {
            outputs[0]->setDataAt<float>(0, 0, h, w,
                                         inputs[0]->dataAt<float>(0, 0, h, w) + weight_.dataAt<float>(0, 0, h, w));
        }
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPUPoEmbedding::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, max_num_, hidden_dim_);
    if (loader.getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        loader.load(&weight_);
    } else {
        weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
    }
    return Op::load(loader);
}

ErrorCode CPUPoEmbedding::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    return Op::free(inputs, outputs);
}

} // namespace mllm
