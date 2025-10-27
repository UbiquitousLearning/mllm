#ifndef OPENCL_ARGSORT_OP_HPP
#define OPENCL_ARGSORT_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLArgSortOp : public Op {
public:
    // 构造函数与CPU版本类似，但不需要threadCount
    OpenCLArgSortOp(Backend *bn, std::string name);
    ~OpenCLArgSortOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    // 内核对象
    cl_kernel kernel_init_indices_ = nullptr;
    cl_kernel kernel_argsort_fp32_ = nullptr;
    cl_kernel kernel_argsort_fp16_ = nullptr;
    cl_kernel kernel_cast_indices_fp32_ = nullptr;
    cl_kernel kernel_cast_indices_fp16_ = nullptr;

    OpenCLBackend *ocl_backend_ = nullptr;
    bool support_fp16_ = false;
};

class OpenCLArgSortOpCreator : public OpenCLBackend::Creator {
public:
    // create 方法与CPU版本类似
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        return new OpenCLArgSortOp(bn, name);
    }
};

} // namespace mllm

#endif // OPENCL_ARGSORT_OP_HPP
