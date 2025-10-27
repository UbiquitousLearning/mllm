#ifndef OPENCL_CLIP_TENSOR_OP_HPP
#define OPENCL_CLIP_TENSOR_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLClipTensorOp : public Op {
public:
    OpenCLClipTensorOp(Backend *bn, std::string name, Chl dim);
    ~OpenCLClipTensorOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    Chl dim_;
    cl_kernel kernel_seq_fp32_ = nullptr;
    cl_kernel kernel_seq_fp16_ = nullptr;
    cl_kernel kernel_dim_fp32_ = nullptr;
    cl_kernel kernel_dim_fp16_ = nullptr;
    OpenCLBackend *ocl_backend_ = nullptr;
};

class OpenCLClipTensorOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        Chl dim = (Chl)op_param.at("dim");
        return new OpenCLClipTensorOp(bn, name, dim);
    }
};

} // namespace mllm

#endif // OPENCL_CLIP_TENSOR_OP_HPP