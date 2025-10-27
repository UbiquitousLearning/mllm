// 文件名: ops/OpenCLSiLUOp.hpp

#ifndef OPENCL_SILU_OP_HPP
#define OPENCL_SILU_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLSiLUOp : public Op {
public:
    OpenCLSiLUOp(Backend *bn, std::string name);
    ~OpenCLSiLUOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    cl_kernel kernel_fp32_ = nullptr;
    cl_kernel kernel_fp16_ = nullptr;

    OpenCLBackend *ocl_backend_ = nullptr;
};

class OpenCLSiLUOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // SiLU通常没有额外参数，但保留 op_param 以备将来扩展
        return new OpenCLSiLUOp(bn, name);
    }
};

} // namespace mllm

#endif // OPENCL_SILU_OP_HPP