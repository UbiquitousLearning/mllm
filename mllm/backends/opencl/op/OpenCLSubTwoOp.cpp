// 文件名: ops/OpenCLSubTwoOp.cpp

#include "OpenCLSubTwoOp.hpp"
#include "Types.hpp"
#include "utils/OpenCLTools.hpp"

namespace mllm {

OpenCLSubTwoOp::OpenCLSubTwoOp(Backend *bn, std::string name) :
    Op(bn, std::move(name)) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    // 关键：加载新的 sub.cl 内核文件
    const std::string kernel_path = "kernel/sub.cl";
    cl_program program = ocl_backend_->getProgram(kernel_path);

    cl_int err;
    // 关键：创建减法内核
    kernel_fp32_buffer_ = clCreateKernel(program, "sub_float", &err);
    check_cl_error(err, "clCreateKernel for sub_float");

    kernel_fp32_image_ = clCreateKernel(program, "sub_float_image2d", &err);
    check_cl_error(err, "clCreateKernel for sub_float_image2d");

    kernel_fp16_buffer_ = clCreateKernel(program, "sub_fp16_vector", &err);
    check_cl_error(err, "clCreateKernel for sub_fp16_vector");

    kernel_fp16_image_ = clCreateKernel(program, "sub_fp16_image2d", &err);
    check_cl_error(err, "clCreateKernel for sub_fp16_image2d");

    sampler_ = clCreateSampler(ocl_backend_->getContext(), CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    check_cl_error(err, "clCreateSampler");
}

OpenCLSubTwoOp::~OpenCLSubTwoOp() {
    if (kernel_fp32_buffer_) clReleaseKernel(kernel_fp32_buffer_);
    if (kernel_fp32_image_) clReleaseKernel(kernel_fp32_image_);
    if (kernel_fp16_buffer_) clReleaseKernel(kernel_fp16_buffer_);
    if (kernel_fp16_image_) clReleaseKernel(kernel_fp16_image_);
    if (sampler_) clReleaseSampler(sampler_);
}

ErrorCode OpenCLSubTwoOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // 形状逻辑与AddTwoOp相同
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    outputs[0]->setDtype(inputs[0]->dtype());
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLSubTwoOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // setUp逻辑与AddTwoOp相同
    for (auto &input : inputs) {
        input->to(MLLM_OPENCL);
    }
    auto output = outputs[0];
    output->setDtype(inputs[0]->dtype());

    auto &out_mem = output->device_memory();
    if (output->dimension() % 4 == 0 && false) {
        out_mem.type = MEM_TYPE_IMAGE_2D;
        out_mem.image_width = output->dimension() / 4;
        out_mem.image_height = output->batch() * output->head() * output->sequence();
    } else {
        out_mem.type = MEM_TYPE_BUFFER;
    }
    output->alloc();
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLSubTwoOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // 这里的逻辑与 AddTwoOp 完全相同，只是 kernel_to_use 指向的是减法内核
    auto input_dtype = inputs[0]->dtype();
    auto output = outputs[0];

    if (output->device_memory().type == MEM_TYPE_IMAGE_2D) {
        cl_kernel kernel_to_use = (input_dtype == MLLM_TYPE_F32) ? kernel_fp32_image_ : kernel_fp16_image_;

        std::vector<Tensor> temp_tensor_storage;
        cl_mem inA_mem = get_image_from_tensor(inputs[0], ocl_backend_, temp_tensor_storage);
        cl_mem inB_mem = get_image_from_tensor(inputs[1], ocl_backend_, temp_tensor_storage);
        cl_mem out_mem_handle = ocl_backend_->get_cl_mem(*output);

        clSetKernelArg(kernel_to_use, 0, sizeof(cl_sampler), &sampler_);
        clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &inA_mem);
        clSetKernelArg(kernel_to_use, 2, sizeof(cl_mem), &inB_mem);
        clSetKernelArg(kernel_to_use, 3, sizeof(cl_mem), &out_mem_handle);

        const int width = static_cast<int>(output->device_memory().image_width);
        const int height = static_cast<int>(output->device_memory().image_height);

        clSetKernelArg(kernel_to_use, 4, sizeof(int), &width);
        clSetKernelArg(kernel_to_use, 5, sizeof(int), &height);

        const size_t global_work_size[2] = {(size_t)width, (size_t)height};
        clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);

    } else {
        cl_kernel kernel_to_use = (input_dtype == MLLM_TYPE_F32) ? kernel_fp32_buffer_ : kernel_fp16_buffer_;

        cl_mem in0_buf = ocl_backend_->get_cl_mem(*inputs[0]);
        cl_mem in1_buf = ocl_backend_->get_cl_mem(*inputs[1]);
        cl_mem out_buf = ocl_backend_->get_cl_mem(*output);

        clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &in0_buf);
        clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &in1_buf);
        clSetKernelArg(kernel_to_use, 2, sizeof(cl_mem), &out_buf);

        size_t count = inputs[0]->count();
        if (input_dtype == MLLM_TYPE_F16) {
            if (count % 4 != 0) {
                throw std::runtime_error("[subTwo]For FP16 vector kernel, tensor count must be a multiple of 4.");
            }
            count /= 4;
        }

        const size_t global_work_size[1] = {count};
        clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    }

    return MLLM_NO_ERROR;
}

} // namespace mllm