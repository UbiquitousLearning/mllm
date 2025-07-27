// opencl/op/OpenCLBinCountOp.hpp

#ifndef OPENCL_BINCOUNT_OP_HPP
#define OPENCL_BINCOUNT_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLBinCountOp : public Op {
public:
    OpenCLBinCountOp(Backend *bn, std::string name);
    ~OpenCLBinCountOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    std::map<std::string, cl_kernel> kernel_map_;
    OpenCLBackend *ocl_backend_ = nullptr;
};

class OpenCLBinCountOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // bincount 通常没有额外参数
        return new OpenCLBinCountOp(bn, name);
    }
};

} // namespace mllm

#endif // OPENCL_BINCOUNT_OP_HPP