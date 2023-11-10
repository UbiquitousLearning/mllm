

#include "CPUKVCache.hpp"

namespace mllm {

CPUKVCache::CPUKVCache(Backend *bn,  string opName, bool multiThread) :
    Op(bn, opName) {
    cache_.setBackend(bn);
    cache_.reshape(0,0,0,0);
}

ErrorCode CPUKVCache::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUKVCache  reshape" << std::endl;
    int a_dim = inputs[0]->dimension();
    int a_sen = inputs[0]->sequence();
    int b_dim = cache_.dimension();
    int b_sen = cache_.sequence();
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), a_sen + b_sen, a_dim);
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUKVCache::load(ParamLoader &loader) {
    //std::cout<<name() << "  CPUKVCache load" << std::endl;
    return Op::load(loader);
}

ErrorCode CPUKVCache::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUKVCache()" << std::endl;
    auto A = inputs[0];
    int a_dim = A->dimension();
    int a_sen = A->sequence();
    int c_sen = outputs[0]->sequence();
    for (int h = 0; h < A->head(); ++h) {
        for (int b = 0; b < A->batch(); ++b) {
            for (int d = 0; d < a_dim; ++d) {
                for (int s = 0; s < c_sen; ++s) {
                    float value = 0;
                    if (s < a_sen) {
                        value = A->dataAt<float>(b, h, s, d);
                    } else {
                        value = cache_.dataAt<float>(b, h, s - a_dim, d);
                    }
                    outputs[0]->setDataAt<float>(b, h, s, d, value);
                }
            }
        }
    }

    cache_.reshape(outputs[0]->shape());
    cache_.alloc();
    cache_.copyFrom(outputs[0]);
    return Op::execute(inputs, outputs);
}

ErrorCode CPUKVCache::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUKVCache() free" << std::endl;
    return Op::free(inputs, outputs);
}
} // namespace mllm

