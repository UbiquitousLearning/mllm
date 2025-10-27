#ifndef OPENCL_DIV_TWO_OP_HPP
#define OPENCL_DIV_TWO_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLDivTwoOp : public Op {
public:
    OpenCLDivTwoOp(Backend *bn, std::string name);
    ~OpenCLDivTwoOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    cl_kernel kernel_fp32_buffer_ = nullptr;
    cl_kernel kernel_fp32_image_ = nullptr;
    cl_kernel kernel_fp16_buffer_ = nullptr; // Note: this kernel is "div_fp16_vector"
    cl_kernel kernel_fp16_scalar_ = nullptr; // Added for non-multiple-of-4 cases
    cl_kernel kernel_fp16_image_ = nullptr;

    cl_sampler sampler_ = nullptr;
    OpenCLBackend *ocl_backend_ = nullptr;
};

class OpenCLDivTwoOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        return new OpenCLDivTwoOp(bn, name);
    }
};

} // namespace mllm

#endif // OPENCL_DIV_TWO_OP_HPP