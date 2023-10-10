#ifndef MLLM_CPUEMBEDDING_HPP
#define MLLM_CPUEMBEDDING_HPP
#include "Op.hpp"
#include "CPUBackend.hpp"
namespace mllm {
class CPUEmbedding final : public Op {
public:
    explicit CPUEmbedding(Backend *bn, int hiddenSize, int vocabSize);
    ~CPUEmbedding() override = default;
    ErrorCode reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
    ErrorCode load(ParamLoader &loader) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;

private:
    Tensor weight_;
    int hiddenSize_;
    int vocabSize_;
};
class CPUEmbeddingCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn) const {
        auto hiddenSize = op_param["hidden_size"];
        auto vocabSize = op_param["vocab_size"];
        return new CPUEmbedding(bn, hiddenSize, vocabSize);
    }
};

} // namespace mllm
#endif // MLLM_CPUEMBEDDING_HPP