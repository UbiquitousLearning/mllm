
#ifndef MLLM_QNNKVCACHE_H
#define MLLM_QNNKVCACHE_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNKVCache : public QNNCommonOp {
public:
    QNNKVCache(Backend *bn, string opName, bool isK);
    virtual ~QNNKVCache() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int cache_size_;
    int dimension_size_;
    bool isK_;
    Tensor seq_pos_;
    Tensor cache_;
    int seq_pos_cpu_;
};

class QNNKVCacheCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNKVCache(bn, name, false);
    }
};

} // namespace mllm

#endif