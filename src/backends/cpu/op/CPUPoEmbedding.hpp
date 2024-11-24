#ifndef MLLM_CPUPOEMBEDDING_H
#define MLLM_CPUPOEMBEDDING_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUPoEmbedding final : public Op {
public:
    CPUPoEmbedding(Backend *bn, string opName, int max_num, int hidden_dim, int threadCount);
    ~CPUPoEmbedding() override = default;
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode load(AbstructLoader &loader) override;
    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
    int max_num_ = 1024;
    int hidden_dim_ = 2048;
    Tensor weight_;
};

class CPUPoEmbeddingCreator : public CPUBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        int max_num = op_param["max_num"];
        int hidden_dim = op_param["hidden_dim"];
        return new CPUPoEmbedding(bn, name, max_num, hidden_dim, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUPOEMBEDDING_H
