// 文件名: ops/OpenCLSubTwoOp.hpp

#ifndef OPENCL_SUBTRACTION_TWO_OP_HPP
#define OPENCL_SUBTRACTION_TWO_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLSubTwoOp : public Op {
public:
    OpenCLSubTwoOp(Backend *bn, std::string name);
    ~OpenCLSubTwoOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    cl_kernel kernel_fp32_buffer_ = nullptr;
    cl_kernel kernel_fp32_image_ = nullptr;
    cl_kernel kernel_fp16_buffer_ = nullptr;
    cl_kernel kernel_fp16_image_ = nullptr;

    cl_sampler sampler_ = nullptr;
    OpenCLBackend *ocl_backend_ = nullptr;
};

// OpenCLSubTwoOp 的创建器
class OpenCLSubTwoOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        return new OpenCLSubTwoOp(bn, name);
    }
};

} // namespace mllm

#endif // OPENCL_SUBTRACTION_TWO_OP_HPP