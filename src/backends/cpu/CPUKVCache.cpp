

#include "CPUKVCache.hpp"
#include "ParamLoader.hpp"

namespace mllm {
CPUKVCache::CPUKVCache(Backend *bn, string opName, int threadCount) : thread_count(threadCount),
    Op(bn, opName) {
    cache_.setBackend(bn);
    cache_.setDtype(MLLM_TYPE_F16);
    cache_limit_ = 500;
}

ErrorCode CPUKVCache::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout<<name() << "  CPUKVCache  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    if(cache_seq_len_ < 0) {
        cache_.reshape(inputs[0]->batch(), inputs[0]->head(), cache_limit_, inputs[0]->dimension());
        cache_.setName(name() + ".Cache");
        cache_.alloc();
        cache_seq_len_ = 0;
    }

    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() + cache_seq_len_, inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUKVCache::load(AbstructLoader &loader) {
    // std::cout<<name() << "  CPUKVCache load" << std::endl;
    return Op::load(loader);
}

ErrorCode CPUKVCache::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout<<name() << "  CPUKVCache()" << std::endl;
    cache_seq_len_ += inputs[0]->sequence();
    return Op::execute(inputs, outputs);
}

ErrorCode CPUKVCache::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout<<name() << "  CPUKVCache() free" << std::endl;
    return Op::free(inputs, outputs);
}


ErrorCode CPUKVCache::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    outputs[0]->setDtype(cache_.dtype());
    outputs[0]->deepCopyFrom(cache_, false, {0,0,cache_seq_len_/cache_limit_,0});
    if (inputs[0]->masterTensor() ==nullptr) {
        inputs[0]->free();
    }
    inputs[0]->deepCopyFrom(cache_, false, {0,0,cache_seq_len_%cache_limit_,0});
#ifdef DEBUG
    std::cout << "*"<<name()<<" setUp*" << std::endl;
#endif
    return MLLM_NO_ERROR;
}
} // namespace mllm
