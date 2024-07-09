/**
 * @file CPUSwaKVCache.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief KV Cache for sliding window attention.
 * @version 0.1
 * @date 2024-05-01
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef MLLM_CPUSWAKVCACHE_H
#define MLLM_CPUSWAKVCACHE_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPUSwaKVCache final : public Op {
public:
    CPUSwaKVCache(Backend *bn, string opName, int n_rep, int window_size, int threadCount);
    ~CPUSwaKVCache() override = default;
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode load(AbstructLoader &loader) override;
    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int n_rep = 1;
    int window_size;
    int thread_count = 4;
    int cache_seq_len = -1;
    int cur_cache_pos = -1;
    Tensor cache;
};

class CPUSwaKVCacheCreator : public CPUBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        int n_rep = (int)op_param["n_rep"];
        int window_size = (int)op_param["window_size"];
        return new CPUSwaKVCache(bn, name, n_rep, window_size, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUSWAKVCACHE_H
