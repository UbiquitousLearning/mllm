// 文件名: ops/OpenCLClipOp.cpp

#include "OpenCLClipOp.hpp"
#include "Types.hpp"
#include "utils/OpenCLTools.hpp" // 包含 check_cl_error
#include <iostream>

namespace mllm {

// 构造函数
OpenCLClipOp::OpenCLClipOp(Backend *bn, std::string name, const std::vector<int> &b, const std::vector<int> &h, const std::vector<int> &s, const std::vector<int> &d) :
    Op(bn, std::move(name)), b_(b), h_(h), s_(s), d_(d) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) {
        throw std::runtime_error("Backend is not OpenCLBackend for OpenCLClipOp");
    }
}

// reshape方法，逻辑与CPU版本完全一致
ErrorCode OpenCLClipOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    int dim_b = inputs[0]->batch();
    int dim_h = inputs[0]->head();
    int dim_s = inputs[0]->sequence();
    int dim_d = inputs[0]->dimension();

    std::vector<std::pair<const std::vector<int> *, int *>> data = {{&b_, &dim_b}, {&h_, &dim_h}, {&s_, &dim_s}, {&d_, &dim_d}};
    for (auto &pair : data) {
        if (pair.first->size() == 2) { // [start, end)
            *pair.second = (*pair.first)[1] - (*pair.first)[0];
        } else if (pair.first->size() == 1) { // [index]
            *pair.second = 1;
        }
    }

    outputs[0]->reshape(dim_b, dim_h, dim_s, dim_d);
    outputs[0]->setDtype(inputs[0]->dtype());
    return MLLM_NO_ERROR;
}

// setUp方法，准备输入输出张量
ErrorCode OpenCLClipOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // 确保输入张量在OpenCL设备上
    inputs[0]->to(MLLM_OPENCL);

    // 根据裁剪参数计算输出形状
    reshape(inputs, outputs);

    // 为输出张量分配设备内存
    outputs[0]->alloc();
    return MLLM_NO_ERROR;
}

// execute方法，执行实际的裁剪操作
ErrorCode OpenCLClipOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    size_t element_size = input->dtypeSize();

    cl_mem in_buf = ocl_backend_->get_cl_mem(*input);
    cl_mem out_buf = ocl_backend_->get_cl_mem(*output);
    cl_command_queue queue = ocl_backend_->getQueue();
    cl_int err;

    // 根据不同的裁剪参数执行相应的拷贝操作
    if (!b_.empty()) {
        // 裁剪 'batch' 维度。这部分数据是连续的，可一次性拷贝。
        int b_start = b_[0];
        if (b_start < 0) b_start += input->batch();
        int b_end = (b_.size() == 2) ? b_[1] : b_start + 1;
        if (b_.size() == 2 && b_end < 0) b_end += input->batch();

        if (b_start < 0 || b_end > input->batch() || b_start >= b_end) {
            return NOT_SUPPORT;
        }
        int count_b = b_end - b_start;

        size_t src_offset_bytes = (size_t)b_start * input->head() * input->sequence() * input->dimension() * element_size;
        size_t copy_size_bytes = (size_t)count_b * input->head() * input->sequence() * input->dimension() * element_size;

        err = clEnqueueCopyBuffer(queue, in_buf, out_buf, src_offset_bytes, 0, copy_size_bytes, 0, nullptr, nullptr);
        check_cl_error(err, "clEnqueueCopyBuffer for batch clipping");

    } else if (!s_.empty()) {
        // ============================ 已修正的逻辑 ============================
        // 裁剪 'sequence' 维度。
        int s_start = s_[0];
        if (s_start < 0) {
            s_start += input->sequence();
        }

        int s_end;
        if (s_.size() == 2) {
            s_end = s_[1];
            if (s_end < 0) {
                s_end += input->sequence();
            }
        } else { // s_.size() == 1
            s_end = s_start + 1;
        }

        // 增加健壮性检查
        if (s_start < 0 || s_end > input->sequence() || s_start >= s_end) {
            std::cerr << "Error: Invalid sequence clip range. Input sequence is " << input->sequence()
                      << ", but calculated range is [" << s_start << ", " << s_end << ")." << std::endl;
            return NOT_SUPPORT;
        }

        int count_s = s_end - s_start;
        // ============================ 修正结束 ============================

        size_t copy_size_per_batch = (size_t)input->head() * count_s * input->dimension() * element_size;
        for (int b = 0; b < input->batch(); ++b) {
            size_t src_offset_bytes = input->offset(b, 0, s_start, 0) * element_size;
            size_t dst_offset_bytes = output->offset(b, 0, 0, 0) * element_size;
            cl_event event;
            err = clEnqueueCopyBuffer(queue, in_buf, out_buf, src_offset_bytes, dst_offset_bytes, copy_size_per_batch, 0, nullptr, &event);
            ocl_backend_->addProfilingEvent(this->name() + "clip", event);
            check_cl_error(err, "clEnqueueCopyBuffer for sequence clipping");
        }

    } else if (!d_.empty()) {
        // 裁剪 'dimension' 维度。这是典型的非连续内存拷贝，使用clEnqueueCopyBufferRect效率最高。
        int d_start = d_[0];
        if (d_start < 0) d_start += input->dimension();

        int d_end;
        if (d_.size() == 2) {
            d_end = d_[1];
            if (d_end < 0) d_end += input->dimension();
        } else { // d_.size() == 1
            d_end = d_start + 1;
        }

        if (d_start < 0 || d_end > input->dimension() || d_start >= d_end) {
            return NOT_SUPPORT;
        }
        int count_d = d_end - d_start;

        // 定义源、目标和区域的3D参数
        size_t src_origin[3] = {(size_t)d_start * element_size, 0, 0}; // X, Y, Z in bytes
        size_t dst_origin[3] = {0, 0, 0};
        size_t region[3] = {(size_t)count_d * element_size, (size_t)input->sequence(), (size_t)(input->batch() * input->head())};

        // 定义内存布局的行间距和切片间距
        size_t src_row_pitch = input->dimension() * element_size;
        size_t src_slice_pitch = input->sequence() * src_row_pitch;
        size_t dst_row_pitch = output->dimension() * element_size;
        size_t dst_slice_pitch = output->sequence() * dst_row_pitch;

        cl_event event;
        err = clEnqueueCopyBufferRect(queue, in_buf, out_buf, src_origin, dst_origin, region,
                                      src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch,
                                      0, nullptr, &event);
        ocl_backend_->addProfilingEvent(this->name(), event);
        check_cl_error(err, "clEnqueueCopyBufferRect for dimension clipping");

    } else {
        std::cerr << "[TODO] OpenCLClipOp does not support this clipping parameter configuration!" << std::endl;
        return NOT_SUPPORT;
    }

    return MLLM_NO_ERROR;
}

// 创建器实现
Op *OpenCLClipOpCreator::create(OpParam op_param, Backend *bn, string name, int threadCount) const {
    // 从op_param中解析出向量参数

    // Example structure: {"b_size": 1, "b_0": 5, "h_size": 0, ...}
    int b_size = op_param.at("b_size");
    int h_size = op_param.at("h_size");
    int s_size = op_param.at("s_size");
    int d_size = op_param.at("d_size");

    std::vector<int> b, h, s, d;
    for (int i = 0; i < b_size; ++i) b.push_back(op_param.at("b_" + std::to_string(i)));
    for (int i = 0; i < h_size; ++i) h.push_back(op_param.at("h_" + std::to_string(i)));
    for (int i = 0; i < s_size; ++i) s.push_back(op_param.at("s_" + std::to_string(i)));
    for (int i = 0; i < d_size; ++i) d.push_back(op_param.at("d_" + std::to_string(i)));

    return new OpenCLClipOp(bn, name, b, h, s, d);
}

} // namespace mllm
