#ifndef OPENCL_MATMUL_OP_HPP
#define OPENCL_MATMUL_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLMatmulOp : public Op {
public:
    OpenCLMatmulOp(Backend *bn, std::string name);
    ~OpenCLMatmulOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    cl_kernel kernel_fp32_ = nullptr;
    cl_kernel kernel_fp16_ = nullptr;
    cl_kernel kernel_fp32_bhsd_ = nullptr;
    cl_kernel kernel_fp16_bhsd_ = nullptr;
    cl_kernel kernel_fp32_transb_ = nullptr;
    cl_kernel kernel_fp16_transb_ = nullptr;
    cl_kernel kernel_fp32_q4_0_transb = nullptr;
    cl_kernel kernel_fp16_q4_0_transb = nullptr;

    OpenCLBackend *ocl_backend_ = nullptr;
    bool use_transb_ = false;
};

class OpenCLMatmulOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        return new OpenCLMatmulOp(bn, name);
    }
};

} // namespace mllm

#endif // OPENCL_MATMUL_OP_HPP