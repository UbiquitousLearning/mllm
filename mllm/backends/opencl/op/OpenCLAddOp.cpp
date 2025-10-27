#include "OpenCLAddOp.hpp"
#include "Types.hpp"
#include "utils/OpenCLTools.hpp"
#include "backends/cpu/third_party/ggml/QuantizeFP16.hpp"

namespace mllm {

OpenCLAddOp::OpenCLAddOp(Backend *bn, std::string name, float data) :
    Op(bn, std::move(name)), data_(data) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    const std::string kernel_path = "kernel/add.cl";
    cl_program program = ocl_backend_->getProgram(kernel_path);

    cl_int err;
    kernel_fp32_buffer_ = clCreateKernel(program, "add_scalar_float", &err);
    check_cl_error(err, "clCreateKernel for add_scalar_float");

    kernel_fp32_image_ = clCreateKernel(program, "add_scalar_float_image2d", &err);
    check_cl_error(err, "clCreateKernel for add_scalar_float_image2d");

    kernel_fp16_buffer_ = clCreateKernel(program, "add_scalar_fp16_vector", &err);
    check_cl_error(err, "clCreateKernel for add_scalar_fp16_vector");

    kernel_fp16_image_ = clCreateKernel(program, "add_scalar_fp16_image2d", &err);
    check_cl_error(err, "clCreateKernel for add_scalar_fp16_image2d");

    sampler_ = clCreateSampler(ocl_backend_->getContext(), CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    check_cl_error(err, "clCreateSampler");
}

OpenCLAddOp::~OpenCLAddOp() {
    if (kernel_fp32_buffer_) clReleaseKernel(kernel_fp32_buffer_);
    if (kernel_fp32_image_) clReleaseKernel(kernel_fp32_image_);
    if (kernel_fp16_buffer_) clReleaseKernel(kernel_fp16_buffer_);
    if (kernel_fp16_image_) clReleaseKernel(kernel_fp16_image_);
    if (sampler_) clReleaseSampler(sampler_);
}

ErrorCode OpenCLAddOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    outputs[0]->setDtype(inputs[0]->dtype());
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLAddOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    inputs[0]->to(MLLM_OPENCL);
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

ErrorCode OpenCLAddOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    /*
    auto input_dtype = inputs[0]->dtype();
    auto output = outputs[0];
    if (output->device_memory().type == MEM_TYPE_IMAGE_2D) {
        // --- Image 优化路径 ---
        cl_kernel kernel_to_use = nullptr;
        if (input_dtype == MLLM_TYPE_F32) {
            kernel_to_use = kernel_fp32_image_;
        } else { // MLLM_TYPE_F16
            kernel_to_use = kernel_fp16_image_;
        }

        std::vector<Tensor> temp_tensor_storage;
        cl_mem inA_mem = get_image_from_tensor(inputs[0], ocl_backend_, temp_tensor_storage);
        cl_mem out_mem_handle = ocl_backend_->get_cl_mem(*output);

        clSetKernelArg(kernel_to_use, 0, sizeof(cl_sampler), &sampler_);
        clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &inA_mem);
        if (input_dtype == MLLM_TYPE_F32) {
            clSetKernelArg(kernel_to_use, 2, sizeof(float), &data_);
        } else { // MLLM_TYPE_F16
            mllm_fp16_t data_fp16 = MLLM_FP32_TO_FP16(data_);
            clSetKernelArg(kernel_to_use, 2, sizeof(mllm_fp16_t), &data_fp16);
        }
        clSetKernelArg(kernel_to_use, 3, sizeof(cl_mem), &out_mem_handle);

        const int width = static_cast<int>(output->device_memory().image_width);
        const int height = static_cast<int>(output->device_memory().image_height);
        clSetKernelArg(kernel_to_use, 4, sizeof(int), &width);
        clSetKernelArg(kernel_to_use, 5, sizeof(int), &height);

        const size_t global_work_size[2] = {(size_t)width, (size_t)height};
        clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);

    } else {
        // --- 普通 Buffer 回退路径 ---
        cl_kernel kernel_to_use = nullptr;
        if (input_dtype == MLLM_TYPE_F32) {
            kernel_to_use = kernel_fp32_buffer_;
        } else { // MLLM_TYPE_F16
            kernel_to_use = kernel_fp16_buffer_;
        }

        cl_mem in0_buf = ocl_backend_->get_cl_mem(*inputs[0]);
        cl_mem out_buf = ocl_backend_->get_cl_mem(*output);

        clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &in0_buf);
        if (input_dtype == MLLM_TYPE_F32) {
            clSetKernelArg(kernel_to_use, 1, sizeof(float), &data_);
        } else { // MLLM_TYPE_F16
            mllm_fp16_t data_fp16 = MLLM_FP32_TO_FP16(data_);
            clSetKernelArg(kernel_to_use, 1, sizeof(mllm_fp16_t), &data_fp16);
        }
        clSetKernelArg(kernel_to_use, 2, sizeof(cl_mem), &out_buf);

        size_t count = inputs[0]->count();
        if (input_dtype == MLLM_TYPE_F16) { count /= 4; }

        const size_t global_work_size[1] = {count};
        clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    }
    */
    auto input = inputs[0];
    auto output = outputs[0];
    // 假设我们决定在这个算子中使用Image路径以获得最佳性能
    bool use_image_path = (output->dimension() % 4 == 0);
    if (use_image_path) {
        // ===================================================
        // ================ Image 优化路径 ===================
        // ===================================================
        // 1. 将输入和输出Tensor原地转换为Image2D类型
        tensorGlobal2Image(*input);
        tensorGlobal2Image(*output); // 注意：这里假设output在setUp中是以Buffer类型分配的
        // 2. 选择内核
        cl_kernel kernel_to_use = nullptr;
        if (input->dtype() == MLLM_TYPE_F32) {
            kernel_to_use = kernel_fp32_image_;
        } else { // MLLM_TYPE_F16
            kernel_to_use = kernel_fp16_image_;
        }
        // 3. 获取Image句柄并设置参数
        cl_mem in_img = ocl_backend_->get_cl_mem(*input);
        cl_mem out_img = ocl_backend_->get_cl_mem(*output);
        clSetKernelArg(kernel_to_use, 0, sizeof(cl_sampler), &sampler_);
        clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &in_img);
        if (input->dtype() == MLLM_TYPE_F32) {
            clSetKernelArg(kernel_to_use, 2, sizeof(float), &data_);
        } else { // MLLM_TYPE_F16
            mllm_fp16_t data_fp16 = MLLM_FP32_TO_FP16(data_);
            clSetKernelArg(kernel_to_use, 2, sizeof(mllm_fp16_t), &data_fp16);
        }
        clSetKernelArg(kernel_to_use, 3, sizeof(cl_mem), &out_img);
        const int width = static_cast<int>(output->device_memory().image_width);
        const int height = static_cast<int>(output->device_memory().image_height);
        clSetKernelArg(kernel_to_use, 4, sizeof(int), &width);
        clSetKernelArg(kernel_to_use, 5, sizeof(int), &height);
        // 4. 执行内核
        const size_t global_work_size[2] = {(size_t)width, (size_t)height};
        clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
        // 5. [重要] 如果后续的算子需要Buffer类型，则将Tensor转换回去
        tensorImage2Global(*input);
        tensorImage2Global(*output);
    } else {
        // ===================================================
        // ============= 普通 Buffer 回退路径 ================
        // ===================================================
        // 确保输入输出都是Buffer（如果之前被转换过）
        tensorImage2Global(*input);
        tensorImage2Global(*output);
        cl_kernel kernel_to_use = nullptr;
        if (input->dtype() == MLLM_TYPE_F32) {
            kernel_to_use = kernel_fp32_buffer_;
        } else { // MLLM_TYPE_F16
            kernel_to_use = kernel_fp16_buffer_;
        }
        cl_mem in0_buf = ocl_backend_->get_cl_mem(*input);
        cl_mem out_buf = ocl_backend_->get_cl_mem(*output);
        clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &in0_buf);
        if (input->dtype() == MLLM_TYPE_F32) {
            clSetKernelArg(kernel_to_use, 1, sizeof(float), &data_);
        } else { // MLLM_TYPE_F16
            mllm_fp16_t data_fp16 = MLLM_FP32_TO_FP16(data_);
            clSetKernelArg(kernel_to_use, 1, sizeof(mllm_fp16_t), &data_fp16);
        }
        clSetKernelArg(kernel_to_use, 2, sizeof(cl_mem), &out_buf);
        size_t count = input->count();
        if (input->dtype() == MLLM_TYPE_F16) { count /= 4; }
        const size_t global_work_size[1] = {count};
        clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    }
    return MLLM_NO_ERROR;
}

} // namespace mllm