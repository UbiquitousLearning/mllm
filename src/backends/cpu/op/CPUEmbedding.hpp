#ifndef MLLM_CPUEMBEDDING_HPP
#define MLLM_CPUEMBEDDING_HPP
#include "Op.hpp"
#include "../CPUBackend.hpp"
#include "Tensor.hpp"
namespace mllm {
class CPUEmbedding final : public Op {
public:
    explicit CPUEmbedding(Backend *bn, string opName, int hiddenSize, int vocabSize, int threadCount);
    ~CPUEmbedding() override = default;
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode load(AbstructLoader &loader) override;
    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    Tensor &weight() {
        return weight_;
    }

private:
    int thread_count = 4;
    Tensor weight_;
    int hiddenSize_;
    int vocabSize_;
};
class CPUEmbeddingCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        auto hiddenSize = op_param["hidden_size"];
        auto vocabSize = op_param["vocab_size"];
        return new CPUEmbedding(bn, name, hiddenSize, vocabSize, threadCount);
    }
};

} // namespace mllm
#endif // MLLM_CPUEMBEDDING_HPP
