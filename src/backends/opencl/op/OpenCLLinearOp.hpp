#ifndef MLLM_OPENCLLINEAROP_H
#define MLLM_OPENCLLINEAROP_H

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLLinearOp final : public Op {
public:
    OpenCLLinearOp(Backend *bn, string opName, int in_features, int out_features, bool bias);
    virtual ~OpenCLLinearOp() override;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int in_features_;
    int out_features_;
    bool support_bias_;

    Tensor weight_;
    Tensor bias_;

    // 使用新的带有 bias 后缀的内核
    cl_kernel kernel_fp32_transb_bias_ = nullptr;
    cl_kernel kernel_fp16_transb_bias_ = nullptr;
    cl_kernel kernel_fp16_q4_0_transb_bias_ = nullptr;
    cl_kernel kernel_fp32_q4_0_transb_bias_ = nullptr;
    cl_kernel kernel_gemv_fp32_q4_0_transb_bias_ = nullptr;        // GEMV
    cl_kernel kernel_gemv_fp16_q4_0_transb_bias_ = nullptr;        // GEMV
    cl_kernel kernel_gemv_fp16_q4_0_transb_bias_half16_ = nullptr; // GEMV for K%16==0

    cl_kernel kernel_fp32_q4_0_transb_bias_image2d_ = nullptr;
    cl_kernel kernel_fp16_q4_0_transb_bias_image2d_ = nullptr;

    cl_kernel kernel_gemv_fp32_q4_0_transb_bias_image2d_ = nullptr;
    cl_kernel kernel_gemv_fp16_q4_0_transb_bias_image2d_ = nullptr;
    OpenCLBackend *ocl_backend_ = nullptr;
};

class OpenCLLinearOpCreator : public OpenCLBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        int in_features = op_param["in_features"];
        int out_features = op_param["out_features"];
        bool bias = op_param["bias"];
        return new OpenCLLinearOp(bn, name, in_features, out_features, bias);
    }
};

} // namespace mllm

#endif // MLLM_OPENCLLINEAROP_H