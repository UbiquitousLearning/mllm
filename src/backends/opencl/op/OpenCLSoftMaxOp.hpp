#ifndef OPENCL_SOFTMAX_OP_HPP
#define OPENCL_SOFTMAX_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLSoftMaxOp : public Op {
public:
    OpenCLSoftMaxOp(Backend *bn, std::string name, int axis, bool do_causal_mask);
    ~OpenCLSoftMaxOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int axis_ = 0;
    bool do_causal_mask_ = false;

    cl_kernel kernel_fp32_d_ = nullptr;
    cl_kernel kernel_fp16_d_ = nullptr;

    OpenCLBackend *ocl_backend_ = nullptr;
};

class OpenCLSoftMaxOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        int axis = op_param["axis"];
        bool do_causal_mask = op_param["do_causal_mask"];
        // threadCount is not used for OpenCL ops
        return new OpenCLSoftMaxOp(bn, name, axis, do_causal_mask);
    }
};

} // namespace mllm

#endif // OPENCL_SOFTMAX_OP_HPP