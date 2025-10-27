#ifndef OPENCL_SUBTRACTION_OP_HPP
#define OPENCL_SUBTRACTION_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLSubOp : public Op {
public:
    OpenCLSubOp(Backend *bn, std::string name, float data);
    ~OpenCLSubOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    float data_; // 用于存储要减去的标量

    cl_kernel kernel_fp32_buffer_ = nullptr;
    cl_kernel kernel_fp32_image_ = nullptr;
    cl_kernel kernel_fp16_buffer_ = nullptr;
    cl_kernel kernel_fp16_image_ = nullptr;

    cl_sampler sampler_ = nullptr;
    OpenCLBackend *ocl_backend_ = nullptr;
};

// OpenCLSubOp 的创建器
class OpenCLSubOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // 从 op_param 中解析出要减去的标量数据
        float data = op_param["data"];
        return new OpenCLSubOp(bn, name, data);
    }
};

} // namespace mllm

#endif // OPENCL_SUBTRACTION_OP_HPP