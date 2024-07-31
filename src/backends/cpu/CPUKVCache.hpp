
#ifndef MLLM_CPUKVCACHE_H
#define MLLM_CPUKVCACHE_H

#include "Op.hpp"
#include "CPUBackend.hpp"
#include "ParamLoader.hpp"

namespace mllm {

class CPUKVCache final : public Op {
public:
    CPUKVCache(Backend *bn, string opName, int n_rep, int cache_max=100, int threadCount=4);
    virtual ~CPUKVCache() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    Tensor cache_;

    int getCacheSeqLen() override{
        return cache_seq_len_;
    }
    void clearCache() override{
        cache_seq_len_ = 0 ;
    }

private:
    int thread_count = 4;

    int cache_seq_len_= -999;
    int n_rep_ = 1;

    int cache_limit_ ;
};

class CPUKVCacheCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int n_rep = (int)op_param["n_rep"];
        int cache_max = (int)op_param["cache_max"];
        return new CPUKVCache(bn, name, n_rep, cache_max, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUKVCACHE_H
