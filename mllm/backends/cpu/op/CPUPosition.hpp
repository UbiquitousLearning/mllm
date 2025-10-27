
#ifndef MLLM_CPUPOSITION_HPP
#define MLLM_CPUPOSITION_HPP

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUPosition final : public Op {
public:
    CPUPosition(Backend *bn, string opName, int threadCount);
    ~CPUPosition() override = default;
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode load(AbstructLoader &loader) override;
    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
    int pos_cnt_;
};

class CPUPositionCreator : public CPUBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new CPUPosition(bn, name, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUPOSITION_HPP
