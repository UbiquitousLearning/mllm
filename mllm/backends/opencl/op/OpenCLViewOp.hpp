#ifndef OPENCL_VIEW_OP_HPP
#define OPENCL_VIEW_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLViewOp : public Op {
public:
    OpenCLViewOp(Backend *bn, std::string name, int b, int h, int s, int d);
    ~OpenCLViewOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int b, h, s, d;
};

// OpenCLViewOp 的创建器
class OpenCLViewOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // 从 op_param 中解析出要减去的标量数据
        int b = static_cast<int>(op_param.at("b"));
        int h = static_cast<int>(op_param.at("h"));
        int s = static_cast<int>(op_param.at("s"));
        int d = static_cast<int>(op_param.at("d"));
        return new OpenCLViewOp(bn, name, b, h, s, d);
    }
};

} // namespace mllm

#endif // OPENCL_VIEW_OP_HPP