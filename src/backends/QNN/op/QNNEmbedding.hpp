
#ifndef MLLM_QNNEMBEDDING_H
#define MLLM_QNNEMBEDDING_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNEmbedding : public QNNCommonOp {
public:
    QNNEmbedding(Backend *bn, string opName, int hiddenSize, int vocabSize);
    virtual ~QNNEmbedding() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    Tensor &weight() {
        return weight_;
    }

private:
    Tensor weight_;
    int hiddenSize_;
    int vocabSize_;
};

class QNNEmbeddingCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        auto hiddenSize = op_param["hidden_size"];
        auto vocabSize = op_param["vocab_size"];
        return new QNNEmbedding(bn, name, hiddenSize, vocabSize);
    }
};

} // namespace mllm

#endif
