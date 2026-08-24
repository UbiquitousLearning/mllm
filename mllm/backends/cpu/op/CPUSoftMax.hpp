#ifndef MLLM_CPUSOFTMAX_H
#define MLLM_CPUSOFTMAX_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUSoftMax final : public Op {
public:
    CPUSoftMax(Backend *bn, string opName, int axis, bool do_causal_mask,
               int threadCount, bool attention_worker = false);
    virtual ~CPUSoftMax() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    ErrorCode executeOnAttentionThread(
        vector<shared_ptr<Tensor>> inputs,
        vector<shared_ptr<Tensor>> outputs);

    int axis_ = 0;
    int thread_count = 4;
    bool do_causal_mask_ = false;
    bool attention_worker_ = false;
};

class CPUSoftMaxCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int axis = op_param["axis"];
        bool do_causal_mask = op_param["do_causal_mask"];
        const auto attention_it = op_param.find("attention_worker");
        return new CPUSoftMax(
            bn, name, axis, do_causal_mask, threadCount,
            attention_it != op_param.end() && attention_it->second != 0.0F);
    }
};
} // namespace mllm

#endif // MLLM_CPUSOFTMAX_H
