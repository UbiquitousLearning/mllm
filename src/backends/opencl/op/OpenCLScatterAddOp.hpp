#ifndef OPENCL_SCATTER_ADD_OP_HPP
#define OPENCL_SCATTER_ADD_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"
#include "Types.hpp"

namespace mllm {

class OpenCLScatterAddOp : public Op {
public:
    OpenCLScatterAddOp(Backend *bn, std::string name, Chl dim);
    ~OpenCLScatterAddOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    Chl dim_; // The axis of `self` that `indices` refers to.
    cl_kernel kernel_fp32_ = nullptr;
    cl_kernel kernel_fp16_ = nullptr;
    OpenCLBackend *ocl_backend_ = nullptr;
};

class OpenCLScatterAddOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // CPU版本的命名是"dim"，我们保持一致
        // Chl dim = (Chl)op_param.at("dim");
        return new OpenCLScatterAddOp(bn, name, SEQUENCE);
    }
};

} // namespace mllm

#endif // OPENCL_SCATTER_ADD_OP_HPP