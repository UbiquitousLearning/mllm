#ifndef OPENCL_TRANSPOSE_FUNC_OP_HPP
#define OPENCL_TRANSPOSE_FUNC_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLTransposeOp : public Op {
public:
    OpenCLTransposeOp(Backend *bn, std::string name, const vector<std::pair<Chl, Chl>> &axiss);
    ~OpenCLTransposeOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    cl_kernel kernel_fp32_2d_ = nullptr;
    cl_kernel kernel_fp16_2d_ = nullptr;

    cl_kernel kernel_fp32_bshd_ = nullptr;
    cl_kernel kernel_fp16_bshd_ = nullptr;

    cl_kernel kernel_fp32_bhsd_ = nullptr;
    cl_kernel kernel_fp16_bhsd_ = nullptr;

    OpenCLBackend *ocl_backend_ = nullptr;
    vector<std::pair<Chl, Chl>> axiss_;
};

class OpenCLTransposeOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // Example: {"num_pairs": 1, "axis1_0": 2, "axis2_0": 1} (HEAD, SEQUENCE)
        int num_pairs = static_cast<int>(op_param.at("num_pairs"));
        vector<std::pair<Chl, Chl>> axiss;
        for (int i = 0; i < num_pairs; ++i) {
            Chl axis1 = (Chl)op_param.at("axis1_" + std::to_string(i));
            Chl axis2 = (Chl)op_param.at("axis2_" + std::to_string(i));
            axiss.push_back({axis1, axis2});
        }
        return new OpenCLTransposeOp(bn, name, axiss);
    }
};

} // namespace mllm

#endif // OPENCL_TRANSPOSE_FUNC_OP_HPP