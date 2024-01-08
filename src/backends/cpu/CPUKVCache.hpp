
#ifndef MLLM_CPUKVCACHE_H
#define MLLM_CPUKVCACHE_H

#include "Op.hpp"
#include "CPUBackend.hpp"
#include "ParamLoader.hpp"

namespace mllm {

class CPUKVCache final : public Op {
public:
    CPUKVCache(Backend *bn, string opName, int threadCount);
    virtual ~CPUKVCache() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    Tensor cache_;

private:
    int thread_count = 4;

    int cache_seq_len_= -999;
    // bool isK_;

    int cache_limit_ ;
};

class CPUKVCacheCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        // bool isK = (bool)op_param["isK"];
        return new CPUKVCache(bn, name, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUKVCACHE_H
