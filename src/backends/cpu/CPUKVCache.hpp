
#ifndef MLLM_CPUKVCACHE_H
#define MLLM_CPUKVCACHE_H

#include "Op.hpp"
#include "CPUBackend.hpp"
#include "ParamLoader.hpp"

namespace mllm {

class CPUKVCache final : public Op {
public:
    CPUKVCache(Backend *bn, string opName, bool multiThread);
    virtual ~CPUKVCache() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    bool support_multi_thread_ = false;
    Tensor cache_;
};

class CPUKVCacheCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new CPUKVCache(bn, name, false);
    }
};

} // namespace mllm

#endif // MLLM_CPUKVCACHE_H
