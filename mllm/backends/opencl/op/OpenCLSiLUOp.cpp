// 文件名: ops/OpenCLSiLUOp.cpp

#include "OpenCLSiLUOp.hpp"
#include "Types.hpp"
#include "utils/OpenCLTools.hpp"
#include <iostream>

namespace mllm {

OpenCLSiLUOp::OpenCLSiLUOp(Backend *bn, std::string name) :
    Op(bn, std::move(name)) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    const std::string kernel_path = "kernel/silu.cl";
    cl_program program = ocl_backend_->getProgram(kernel_path);

    cl_int err;
    kernel_fp32_ = clCreateKernel(program, "silu_fp32", &err);
    check_cl_error(err, "clCreateKernel for silu_fp32");

    kernel_fp16_ = clCreateKernel(program, "silu_fp16", &err);
    check_cl_error(err, "clCreateKernel for silu_fp16");
}

OpenCLSiLUOp::~OpenCLSiLUOp() {
    if (kernel_fp32_) clReleaseKernel(kernel_fp32_);
    if (kernel_fp16_) clReleaseKernel(kernel_fp16_);
}

ErrorCode OpenCLSiLUOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // SiLU 是元素级操作，输出形状与输入形状完全相同
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLSiLUOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // 确保输入在设备上，并为输出分配内存
    inputs[0]->to(MLLM_OPENCL);
    outputs[0]->setDtype(inputs[0]->dtype());
    outputs[0]->alloc();
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLSiLUOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    const int count = input->count();

    cl_kernel kernel_to_use = nullptr;
    size_t global_work_size[1];
    const size_t *local_work_size_ptr = nullptr;
    const size_t local_work_size[1] = {256}; // 典型的工作组大小

    if (input->dtype() == MLLM_TYPE_F32) {
        kernel_to_use = kernel_fp32_;
        global_work_size[0] = count;
    } else if (input->dtype() == MLLM_TYPE_F16) {
        kernel_to_use = kernel_fp16_;

        // ✨ **核心修正**: 简化并修正启动配置
        // 我们只需要启动足够的线程来处理向量部分
        size_t vec_count = count / 4;
        // 向上取整到工作组大小的倍数
        global_work_size[0] = ((vec_count + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0];
        // 如果元素总数少于一个工作组的大小，确保至少启动一个工作组
        if (global_work_size[0] == 0 && count > 0) {
            global_work_size[0] = local_work_size[0];
        }
        local_work_size_ptr = local_work_size;

    } else {
        return NOT_SUPPORT;
    }

    cl_mem in_buf = ocl_backend_->get_cl_mem(*input);
    cl_mem out_buf = ocl_backend_->get_cl_mem(*output);

    clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &in_buf);
    clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &out_buf);
    clSetKernelArg(kernel_to_use, 2, sizeof(int), &count);

    cl_event event;
    cl_int err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 1, nullptr, global_work_size, local_work_size_ptr, 0, nullptr, &event);
    ocl_backend_->addProfilingEvent(this->name(), event);
    check_cl_error(err, "clEnqueueNDRangeKernel for SiLU");

    return MLLM_NO_ERROR;
}

} // namespace mllm