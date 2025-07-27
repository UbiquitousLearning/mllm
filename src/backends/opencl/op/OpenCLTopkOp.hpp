#ifndef OPENCL_TOPK_OP_HPP
#define OPENCL_TOPK_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLTopkOp : public Op {
public:
    OpenCLTopkOp(Backend *bn, std::string name, int k, Chl dim);
    ~OpenCLTopkOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int k_;
    Chl dim_;

    cl_kernel kernel_fp32_ = nullptr;
    cl_kernel kernel_fp16_ = nullptr;
    OpenCLBackend *ocl_backend_ = nullptr;
};

class OpenCLTopkOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        int k = static_cast<int>(op_param.at("k"));
        Chl dim = (Chl)op_param.at("dim");
        return new OpenCLTopkOp(bn, name, k, dim);
    }
};

} // namespace mllm

#endif // OPENCL_TOPK_OP_HPP