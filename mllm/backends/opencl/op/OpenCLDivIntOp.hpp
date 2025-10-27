#ifndef OPENCL_DIV_INT_OP_HPP
#define OPENCL_DIV_INT_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLDivIntOp : public Op {
public:
    OpenCLDivIntOp(Backend *bn, std::string name, float data);
    ~OpenCLDivIntOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    float data_; // 用于存储要除的标量

    cl_kernel kernel_fp32_buffer_ = nullptr;
    cl_kernel kernel_fp32_image_ = nullptr;
    cl_kernel kernel_fp16_buffer_ = nullptr; // Vectorized version
    cl_kernel kernel_fp16_image_ = nullptr;
    cl_kernel kernel_fp16_buffer_scalar_ = nullptr; // ADDED: Scalar (element-wise) version

    cl_sampler sampler_ = nullptr;
    OpenCLBackend *ocl_backend_ = nullptr;
};

class OpenCLDivIntOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        float data = op_param["data"];
        return new OpenCLDivIntOp(bn, name, data);
    }
};

} // namespace mllm

#endif // OPENCL_DIV_INT_OP_HPP