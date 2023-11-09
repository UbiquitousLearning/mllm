#ifndef MLLM_NNAPI_COMMON_OP_H
#define MLLM_NNAPI_COMMON_OP_H

#include "Op.hpp"
#include "NNAPIBackend.hpp"
#include <cstdint>
#include <vector>

namespace mllm {

class NNAPICommonOp : public Op {
public:
    NNAPICommonOp(Backend *bn, string name);
    virtual ~NNAPICommonOp() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override = 0;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(ParamLoader &loader) override;

protected:
    NNAPIBackend *nnapiBackend_;
    std::vector<uint32_t> getTensorIdxs(const vector<shared_ptr<Tensor>> &tensors);
    uint32_t getTensorIdx(const Tensor *t, bool isReshape = false, std::vector<uint32_t> dims = {});
    template <typename T>
    inline uint32_t buildScalar(T scalar) {
        return nnapiBackend_->buildScalar(scalar);
    }
    uint32_t buildConstant(const void *data, size_t size, OperandCode dtype, std::vector<uint32_t> dims = {}, const float *scales = nullptr, int zero = 0);
    uint32_t buildTensor(OperandCode dtype, std::vector<int> dims);
    ErrorCode buildOperation(int op, const std::vector<uint32_t> &inputs, const std::vector<uint32_t> &outputs);
    int formatAxis(int axis, const Tensor *t);
};
} // namespace mllm

#endif // MLLM_NNAPI_COMMON_OP_H