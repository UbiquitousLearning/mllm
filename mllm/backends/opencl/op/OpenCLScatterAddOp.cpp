#include "OpenCLScatterAddOp.hpp"
#include "Types.hpp"
// #include "utils/OpenCLTools.hpp"

namespace mllm {

// 构造函数、析构函数、reshape、setUp 保持不变...
OpenCLScatterAddOp::OpenCLScatterAddOp(Backend *bn, std::string name, Chl dim) :
    Op(bn, std::move(name)), dim_(dim) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    const std::string kernel_path = "kernel/scatter_add.cl";
    // cl_program program = ocl_backend_->getProgram(kernel_path, "-cl-std=CL1.2");
    std::string build_options;
    if (ocl_backend_->has_fp16_support()) {
        build_options += " -DSUPPORTS_FP16";
    }

    cl_program program = ocl_backend_->getProgram(kernel_path, build_options + " -cl-std=CL1.2");

    cl_int err;
    kernel_fp32_ = clCreateKernel(program, "scatter_add_fp32", &err);
    check_cl_error(err, "clCreateKernel for scatter_add_fp32");
    kernel_fp16_ = clCreateKernel(program, "scatter_add_fp16", &err);
    check_cl_error(err, "clCreateKernel for scatter_add_fp16");
}

OpenCLScatterAddOp::~OpenCLScatterAddOp() {
    if (kernel_fp32_) clReleaseKernel(kernel_fp32_);
    if (kernel_fp16_) clReleaseKernel(kernel_fp16_);
}

ErrorCode OpenCLScatterAddOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLScatterAddOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    inputs[0]->to(MLLM_OPENCL);
    inputs[1]->to(MLLM_OPENCL);
    inputs[2]->to(MLLM_OPENCL);
    return MLLM_NO_ERROR;
}

// **execute 函数已修正**
ErrorCode OpenCLScatterAddOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto self = inputs[0];
    auto value = inputs[1];
    auto indices = inputs[2];

    if (dim_ != SEQUENCE) {
        std::cerr << "This version of OpenCLScatterAddOp only supports SEQUENCE dimension." << std::endl;
        return NOT_SUPPORT;
    }
    if (self->ctype() != BSHD || value->ctype() != BSHD) {
        return NOT_SUPPORT;
    }

    cl_kernel kernel_to_use;
    if (self->dtype() == MLLM_TYPE_F32) {
        kernel_to_use = kernel_fp32_;
    } else if (self->dtype() == MLLM_TYPE_F16) {
        kernel_to_use = kernel_fp16_;
    } else {
        return NOT_SUPPORT;
    }

    cl_mem self_buf = ocl_backend_->get_cl_mem(*self);
    cl_mem value_buf = ocl_backend_->get_cl_mem(*value);
    cl_mem indices_buf = ocl_backend_->get_cl_mem(*indices);

    const int B = self->batch();
    const int H = self->head();
    const int D = self->dimension();
    const int S_self = self->sequence();
    const int S_value = value->sequence();

    clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &self_buf);    // 参数 0: self
    clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &value_buf);   // 参数 1: value
    clSetKernelArg(kernel_to_use, 2, sizeof(cl_mem), &indices_buf); // 参数 2: indices

    clSetKernelArg(kernel_to_use, 3, sizeof(int), &B);
    clSetKernelArg(kernel_to_use, 4, sizeof(int), &H);
    clSetKernelArg(kernel_to_use, 5, sizeof(int), &D);
    clSetKernelArg(kernel_to_use, 6, sizeof(int), &S_self);
    clSetKernelArg(kernel_to_use, 7, sizeof(int), &S_value);
    const size_t global_work_size[3] = {(size_t)D, (size_t)H, (size_t)B * S_value};

    cl_event event;
    cl_int err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 3, nullptr, global_work_size, nullptr, 0, nullptr, &event);
    check_cl_error(err, "clEnqueueNDRangeKernel for ScatterAdd (in-place, sequence)");

    return MLLM_NO_ERROR;
}

} // namespace mllm