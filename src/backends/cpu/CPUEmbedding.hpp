#ifndef MLLM_CPUEMBEDDING_HPP
#define MLLM_CPUEMBEDDING_HPP
#include "Op.hpp"
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

} // namespace mllm
#endif // MLLM_CPUEMBEDDING_HPP
