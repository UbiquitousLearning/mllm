
#ifndef MLLM_CPUKVCACHENPU_H
#define MLLM_CPUKVCACHENPU_H

#include "Op.hpp"
#include "../CPUBackend.hpp"
#include "ParamLoader.hpp"

namespace mllm {
/**
 * @brief KVCache operator for NPU-CPU co processing
 *
 * The output of NPU linear is a continuous memory block, which is not suitable for Non Copy cache when the channel is BHDS.
 * For input tensor with channel of BSHD, the input & output will be a sub-tensor of the cache tensor.
 * For input tensor with channel of BHDS, the input will be copied to the cache tensor.
 */
class CPUKVCacheNPU final : public Op {
public:
    CPUKVCacheNPU(Backend *bn, string opName, int n_rep, int cache_max = 100, int threadCount = 4);
    virtual ~CPUKVCacheNPU() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    Tensor cache_;

    int getCacheSeqLen() override {
        return cache_seq_len_;
    }
    void clearCache() override {
        cache_seq_len_ = 0;
    }

private:
    int thread_count = 4;

    int cache_seq_len_ = -999;
    int n_rep_ = 1;
    bool isDecoding = false;

    int cache_limit_;
};

class CPUKVCacheNPUCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int n_rep = (int)op_param["n_rep"] == 0 ? 1 : (int)op_param["n_rep"];
        int cache_max = (int)op_param["cache_max"];
        return new CPUKVCacheNPU(bn, name, n_rep, cache_max, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUKVCACHE_H