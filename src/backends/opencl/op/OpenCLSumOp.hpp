#ifndef OPENCL_SUM_OP_HPP
#define OPENCL_SUM_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLSumOp : public Op {
public:
    OpenCLSumOp(Backend *bn, std::string name, Chl axis);
    ~OpenCLSumOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    Chl axis_;
    cl_kernel kernel_fp32_ = nullptr;
    cl_kernel kernel_fp16_ = nullptr;
    OpenCLBackend *ocl_backend_ = nullptr;
};

class OpenCLSumOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        Chl axis = (Chl)op_param.at("dim");
        return new OpenCLSumOp(bn, name, axis);
    }
};

} // namespace mllm

#endif // OPENCL_SUM_OP_HPP