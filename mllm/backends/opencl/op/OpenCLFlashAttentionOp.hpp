#ifndef OPENCL_FLASHATTENTION_OP_HPP
#define OPENCL_FLASHATTENTION_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLFlashAttentionOp : public Op {
public:
    OpenCLFlashAttentionOp(Backend *bn, std::string name, bool causal_mask);
    ~OpenCLFlashAttentionOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    bool causal_mask_;
    cl_kernel kernel_fp32_ = nullptr;
    cl_kernel kernel_fp32_decode_ = nullptr;
    cl_kernel kernel_fp16_ = nullptr;
    cl_kernel kernel_fp16_decode_ = nullptr;
    OpenCLBackend *ocl_backend_ = nullptr;

    cl_kernel kernel_fp32_image_ = nullptr;
    cl_kernel kernel_fp32_decode_image_ = nullptr;
    cl_kernel kernel_fp16_image_ = nullptr;
    cl_kernel kernel_fp16_decode_image_ = nullptr;
    cl_sampler sampler_ = nullptr;
};

class OpenCLFlashAttentionOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        bool causal_mask = (bool)op_param.at("causal_mask");
        return new OpenCLFlashAttentionOp(bn, name, causal_mask);
    }
};

} // namespace mllm

#endif // OPENCL_FLASHATTENTION_OP_HPP
