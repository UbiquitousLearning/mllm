#ifndef OPENCL_SPLIT_OP_HPP
#define OPENCL_SPLIT_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"
#include <vector>

namespace mllm {

class OpenCLSplitOp : public Op {
public:
    OpenCLSplitOp(Backend *bn, std::string name, int num_splits, const std::vector<int> &each_dims, Chl split_dim, int head_size = 1);
    ~OpenCLSplitOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int num_splits_;
    std::vector<int> each_dims_;
    int head_size_ = 1;
    Chl split_dim_;

    cl_kernel kernel_fp32_ = nullptr;
    cl_kernel kernel_fp16_ = nullptr;
    OpenCLBackend *ocl_backend_ = nullptr;
};

class OpenCLSplitOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        int num_splits = static_cast<int>(op_param.at("num_splits"));
        std::vector<int> each_dims;
        for (int i = 0; i < num_splits; ++i) {
            each_dims.push_back(static_cast<int>(op_param.at("dim_" + std::to_string(i))));
        }
        Chl split_dim = (Chl)op_param.at("split_dim");
        int head_size = static_cast<int>(op_param.at("head_size"));
        return new OpenCLSplitOp(bn, name, num_splits, each_dims, split_dim, head_size);
    }
};

} // namespace mllm

#endif // OPENCL_SPLIT_OP_HPP