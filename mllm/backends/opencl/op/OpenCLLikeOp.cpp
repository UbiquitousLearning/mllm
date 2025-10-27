#include "OpenCLLikeOp.hpp"
#include "Types.hpp"
#include "utils/OpenCLTools.hpp"

namespace mllm {

OpenCLLikeOp::OpenCLLikeOp(Backend *bn, std::string name, float like_value) :
    Op(bn, std::move(name)), like_value_(like_value) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    const std::string kernel_path = "kernel/like.cl";
    cl_program program = ocl_backend_->getProgram(kernel_path);

    cl_int err;
    // 内核将处理所有数据类型，但在内核内部进行转换
    kernel_ = clCreateKernel(program, "like", &err);
    check_cl_error(err, "clCreateKernel for like");
}

OpenCLLikeOp::~OpenCLLikeOp() {
    if (kernel_) clReleaseKernel(kernel_);
}

ErrorCode OpenCLLikeOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // 输出的形状和数据类型与输入完全一致
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    outputs[0]->setDtype(inputs[0]->dtype());
    outputs[0]->to(MLLM_OPENCL); // 确保输出张量在OpenCL上
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLLikeOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // Like 操作不需要输入张量的数据，所以 inputs[0] 无需 to(MLLM_OPENCL)
    outputs[0]->setDtype(inputs[0]->dtype());
    outputs[0]->alloc();
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLLikeOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto output = outputs[0];
    cl_mem out_buf = ocl_backend_->get_cl_mem(*output);
    const int count = output->count();
    const int dtype_size = output->dtypeSize();

    clSetKernelArg(kernel_, 0, sizeof(cl_mem), &out_buf);
    clSetKernelArg(kernel_, 1, sizeof(float), &like_value_);
    clSetKernelArg(kernel_, 2, sizeof(int), &count);
    clSetKernelArg(kernel_, 3, sizeof(int), &dtype_size);

    const size_t global_work_size = (size_t)count;
    cl_event event;
    cl_int err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_, 1, nullptr,
                                        &global_work_size, nullptr, 0, nullptr, &event);
    check_cl_error(err, "clEnqueueNDRangeKernel for Like");

    return MLLM_NO_ERROR;
}

} // namespace mllm