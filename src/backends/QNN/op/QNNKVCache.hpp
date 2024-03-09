
#ifndef MLLM_QNNKVCACHE_H
#define MLLM_QNNKVCACHE_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNKVCache : public QNNCommonOp {
public:
    QNNKVCache(Backend *bn, string opName, int cache_max);
    virtual ~QNNKVCache() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int cache_size_;
    int dimension_size_;
    bool isK_;
    Tensor seq_pos_;
    Tensor cache_;
    int seq_pos_cpu_;

    // each op has three size
    // 1. op reshape size -> CPU execution size.
    // 2. op alloc size -> input and output memory buffer.
    // 3. op qnn size -> qnn op execution size.

    std::vector<uint> alloc_size_;
    std::vector<uint> qnn_size_;


};

class QNNKVCacheCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNKVCache(bn, name, (int)op_param["cache_max"]);
    }
};

} // namespace mllm

#endif
