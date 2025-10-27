// 文件名: ops/OpenCLClipOp.hpp

#ifndef OPENCL_CLIP_OP_HPP
#define OPENCL_CLIP_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"
#include <vector>

namespace mllm {

/**
 * @brief OpenCL实现的Clip操作，用于裁剪张量的部分区域。
 */
class OpenCLClipOp : public Op {
public:
    /**
     * @brief 构造函数
     * @param bn 后端指针
     * @param name 操作名
     * @param b 裁剪batch维度的参数
     * @param h 裁剪head维度的参数
     * @param s 裁剪sequence维度的参数
     * @param d 裁剪dimension维度的参数
     */
    OpenCLClipOp(Backend *bn, std::string name, const std::vector<int> &b, const std::vector<int> &h, const std::vector<int> &s, const std::vector<int> &d);

    // 默认析构函数即可
    ~OpenCLClipOp() override = default;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    // 存储各个维度的裁剪参数
    std::vector<int> b_;
    std::vector<int> h_;
    std::vector<int> s_;
    std::vector<int> d_;

    OpenCLBackend *ocl_backend_ = nullptr;
};

/**
 * @brief OpenCLClipOp的创建器类，用于工厂模式创建实例。
 */
class OpenCLClipOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override;
};

} // namespace mllm

#endif // OPENCL_CLIP_OP_HPP
