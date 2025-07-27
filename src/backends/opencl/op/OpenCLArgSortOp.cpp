#include "OpenCLArgSortOp.hpp"
#include "Types.hpp"
// #include "utils/OpenCLTools.hpp"
#include <cmath>
#include <string> // For error message formatting

namespace mllm {

// 构造函数、析构函数、reshape 和 setUp 保持不变...
OpenCLArgSortOp::OpenCLArgSortOp(Backend *bn, std::string name) :
    Op(bn, std::move(name)) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    support_fp16_ = ocl_backend_->has_fp16_support();
    const std::string kernel_path = "kernel/argsort.cl";

    std::string build_options;
    if (support_fp16_) {
        build_options += " -DSUPPORTS_FP16";
    }
    cl_program program = ocl_backend_->getProgram(kernel_path, build_options);

    cl_int err;
    kernel_init_indices_ = clCreateKernel(program, "init_indices", &err);
    check_cl_error(err, "clCreateKernel init_indices");
    kernel_argsort_fp32_ = clCreateKernel(program, "bitonic_argsort_step_fp32", &err);
    check_cl_error(err, "clCreateKernel bitonic_argsort_step_fp32");
    kernel_cast_indices_fp32_ = clCreateKernel(program, "cast_indices_to_fp32", &err);
    check_cl_error(err, "clCreateKernel cast_indices_to_fp32");

    kernel_argsort_fp16_ = clCreateKernel(program, "bitonic_argsort_step_fp16", &err);
    check_cl_error(err, "clCreateKernel bitonic_argsort_step_fp16");
    kernel_cast_indices_fp16_ = clCreateKernel(program, "cast_indices_to_fp16", &err);
    check_cl_error(err, "clCreateKernel cast_indices_to_fp16");
}

OpenCLArgSortOp::~OpenCLArgSortOp() {
    if (kernel_init_indices_) clReleaseKernel(kernel_init_indices_);
    if (kernel_argsort_fp32_) clReleaseKernel(kernel_argsort_fp32_);
    if (kernel_cast_indices_fp32_) clReleaseKernel(kernel_cast_indices_fp32_);
    if (kernel_argsort_fp16_) clReleaseKernel(kernel_argsort_fp16_);
    if (kernel_cast_indices_fp16_) clReleaseKernel(kernel_cast_indices_fp16_);
}

ErrorCode OpenCLArgSortOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1 && outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    outputs[0]->setDtype(inputs[0]->dtype());
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLArgSortOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    inputs[0]->to(MLLM_OPENCL);
    outputs[0]->alloc();
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLArgSortOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto output = outputs[0];

    const int batch_size = input->batch() * input->head() * input->sequence();
    const int N = input->dimension();
    cl_int err;

    const size_t input_bytes = (size_t)input->count() * input->dtypeSize();
    cl_mem input_buf = ocl_backend_->get_cl_mem(*input);
    cl_mem output_buf = ocl_backend_->get_cl_mem(*output);

    // 1. 创建临时 buffer
    cl_mem temp_values_buf = clCreateBuffer(ocl_backend_->getContext(), CL_MEM_READ_WRITE, input_bytes, nullptr, &err);
    check_cl_error(err, "clCreateBuffer for temp_values");
    cl_mem indices_buf = clCreateBuffer(ocl_backend_->getContext(), CL_MEM_READ_WRITE, (size_t)batch_size * N * sizeof(int), nullptr, &err);
    check_cl_error(err, "clCreateBuffer for indices");

    // 2. 在设备上复制数据到临时 buffer
    err = clEnqueueCopyBuffer(ocl_backend_->getQueue(), input_buf, temp_values_buf, 0, 0, input_bytes, 0, nullptr, nullptr);
    check_cl_error(err, "clEnqueueCopyBuffer to temp_values_buf");

    // 3. 初始化索引 buffer
    size_t global_work_size_init = (size_t)batch_size * N;
    clSetKernelArg(kernel_init_indices_, 0, sizeof(cl_mem), &indices_buf);
    clSetKernelArg(kernel_init_indices_, 1, sizeof(int), &N);
    err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_init_indices_, 1, nullptr, &global_work_size_init, nullptr, 0, nullptr, nullptr);
    check_cl_error(err, "clEnqueueNDRangeKernel init_indices");

    // 4. 执行 Bitonic Sort
    cl_kernel kernel_sort = (input->dtype() == MLLM_TYPE_F32) ? kernel_argsort_fp32_ : kernel_argsort_fp16_;
    int descending = 0; // 0 for ascending

    int power_of_2_N = 1;
    while (power_of_2_N < N) {
        power_of_2_N <<= 1;
    }
    int num_stages = (N > 1) ? std::log2(power_of_2_N) : 0;

    for (int stage = 0; stage < num_stages; ++stage) {
        for (int pass = stage; pass >= 0; --pass) {
            size_t global_work_size_sort[2] = {(size_t)power_of_2_N / 2, (size_t)batch_size};

            clSetKernelArg(kernel_sort, 0, sizeof(cl_mem), &temp_values_buf);
            clSetKernelArg(kernel_sort, 1, sizeof(cl_mem), &indices_buf);
            clSetKernelArg(kernel_sort, 2, sizeof(int), &N);
            clSetKernelArg(kernel_sort, 3, sizeof(int), &stage);
            clSetKernelArg(kernel_sort, 4, sizeof(int), &pass);
            clSetKernelArg(kernel_sort, 5, sizeof(int), &descending);

            err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_sort, 2, nullptr, global_work_size_sort, nullptr, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                std::string error_msg = "clEnqueueNDRangeKernel bitonic_sort failed with code " + std::to_string(err)
                                        + " at stage " + std::to_string(stage) + ", pass " + std::to_string(pass);
                throw std::runtime_error(error_msg);
            }
        }
    }

    // 5. 转换索引类型并写入输出
    cl_kernel kernel_cast = (output->dtype() == MLLM_TYPE_F32) ? kernel_cast_indices_fp32_ : kernel_cast_indices_fp16_;
    size_t global_work_size_cast = (size_t)batch_size * N;
    clSetKernelArg(kernel_cast, 0, sizeof(cl_mem), &indices_buf);
    clSetKernelArg(kernel_cast, 1, sizeof(cl_mem), &output_buf);
    err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_cast, 1, nullptr, &global_work_size_cast, nullptr, 0, nullptr, nullptr);
    check_cl_error(err, "clEnqueueNDRangeKernel cast_indices");

    // 6. 释放临时 buffer
    clReleaseMemObject(temp_values_buf);
    clReleaseMemObject(indices_buf);

    return MLLM_NO_ERROR;
}

} // namespace mllm
