#ifndef OPENCL_LIKE_OP_HPP
#define OPENCL_LIKE_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLLikeOp : public Op {
public:
    OpenCLLikeOp(Backend *bn, std::string name, float like_value);
    ~OpenCLLikeOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    float like_value_;
    cl_kernel kernel_ = nullptr;
    OpenCLBackend *ocl_backend_ = nullptr;
};

class OpenCLLikeOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // 从 op_param 中获取 "like_value"
        float like_value = op_param.at("like_value");
        return new OpenCLLikeOp(bn, name, like_value);
    }
};

} // namespace mllm

#endif // OPENCL_LIKE_OP_HPP