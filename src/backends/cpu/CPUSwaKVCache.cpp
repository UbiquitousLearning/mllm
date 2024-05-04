/**
 * @file CPUSwaKVCache.cpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-05-01
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "CPUSwaKVCache.hpp"

namespace mllm {

CPUSwaKVCache::CPUSwaKVCache(Backend *bn, string opName, int n_rep, int window_size, int threadCount) :
    n_rep(n_rep),
    window_size(window_size),
    thread_count(threadCount),
    Op(bn, opName) {
    cache.setBackend(bn);
    cache.setDtype(MLLM_TYPE_F16);
}

ErrorCode CPUSwaKVCache::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    if (cache_seq_len < 0) {
        cache.reshape(inputs[0]->batch(), inputs[0]->head() * n_rep, window_size, inputs[0]->dimension());
        cache.setName(name() + ".Cache");
        cache.alloc();
        cache_seq_len = 0;
    }

    int sequence_len = (inputs[0]->sequence() + cache_seq_len) > window_size ? window_size : (inputs[0]->sequence() + cache_seq_len);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head() * n_rep, sequence_len, inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUSwaKVCache::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::execute(inputs, outputs);
}

ErrorCode CPUSwaKVCache::load(AbstructLoader &loader) {
    return Op::load(loader);
}

ErrorCode CPUSwaKVCache::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}

ErrorCode CPUSwaKVCache::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::setUp(inputs, outputs);
}
} // namespace mllm
