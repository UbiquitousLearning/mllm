#ifndef OPENCL_RMSNORM_OP_HPP
#define OPENCL_RMSNORM_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLRMSNormOp : public Op {
public:
    OpenCLRMSNormOp(Backend *bn, std::string name, int normSize, float epsilon = 1e-6, bool add_unit_offset_ = false);
    ~OpenCLRMSNormOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode load(AbstructLoader &loader) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    float epsilon_;
    Tensor weight_;
    int normSize_;
    bool add_unit_offset_;

    cl_kernel kernel_fp32_ = nullptr;
    cl_kernel kernel_fp16_ = nullptr;
    OpenCLBackend *ocl_backend_ = nullptr;
};

class OpenCLRMSNormOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        int normSize = (int)op_param["norm_size"];
        float epsilon = op_param.count("epsilon") ? (float)op_param["epsilon"] : 1e-6f;
        bool add_unit_offset = op_param.count("add_unit_offset") ? (bool)op_param["add_unit_offset"] : false;
        return new OpenCLRMSNormOp(bn, name, normSize, epsilon, add_unit_offset);
    }
};

} // namespace mllm

#endif // OPENCL_RMSNORM_OP_HPP