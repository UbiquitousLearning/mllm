#include "OpenCLSubOp.hpp"
#include "Types.hpp"
#include "utils/OpenCLTools.hpp"
#include "backends/cpu/third_party/ggml/QuantizeFP16.hpp"

namespace mllm {

// 构造函数：与AddOp类似，但内核名称是硬编码的，所以我们仍然指向 "add.cl"
OpenCLSubOp::OpenCLSubOp(Backend *bn, std::string name, float data) :
    Op(bn, std::move(name)), data_(data) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    // 复用 add.cl 内核文件
    const std::string kernel_path = "kernel/add.cl";
    cl_program program = ocl_backend_->getProgram(kernel_path);

    cl_int err;
    // 注意：我们创建的是 add_scalar_* 内核，因为我们将在内核中通过数学技巧实现减法
    kernel_fp32_buffer_ = clCreateKernel(program, "add_scalar_float", &err);
    check_cl_error(err, "clCreateKernel for add_scalar_float (used by Sub)");

    kernel_fp32_image_ = clCreateKernel(program, "add_scalar_float_image2d", &err);
    check_cl_error(err, "clCreateKernel for add_scalar_float_image2d (used by Sub)");

    kernel_fp16_buffer_ = clCreateKernel(program, "add_scalar_fp16_vector", &err);
    check_cl_error(err, "clCreateKernel for add_scalar_fp16_vector (used by Sub)");

    kernel_fp16_image_ = clCreateKernel(program, "add_scalar_fp16_image2d", &err);
    check_cl_error(err, "clCreateKernel for add_scalar_fp16_image2d (used by Sub)");

    sampler_ = clCreateSampler(ocl_backend_->getContext(), CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    check_cl_error(err, "clCreateSampler");
}

OpenCLSubOp::~OpenCLSubOp() {
    if (kernel_fp32_buffer_) clReleaseKernel(kernel_fp32_buffer_);
    if (kernel_fp32_image_) clReleaseKernel(kernel_fp32_image_);
    if (kernel_fp16_buffer_) clReleaseKernel(kernel_fp16_buffer_);
    if (kernel_fp16_image_) clReleaseKernel(kernel_fp16_image_);
    if (sampler_) clReleaseSampler(sampler_);
}

ErrorCode OpenCLSubOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    outputs[0]->setDtype(inputs[0]->dtype());
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLSubOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // 与 AddOp 逻辑相同
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

ErrorCode OpenCLSubOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input_dtype = inputs[0]->dtype();
    auto output = outputs[0];

    // ✨ 核心技巧：A - B  等价于 A + (-B)
    // 我们将要减去的标量取反，然后调用加法内核
    float neg_data = -data_;

    if (output->device_memory().type == MEM_TYPE_IMAGE_2D) {
        // Image 路径
        cl_kernel kernel_to_use = (input_dtype == MLLM_TYPE_F32) ? kernel_fp32_image_ : kernel_fp16_image_;
        std::vector<Tensor> temp_tensor_storage;
        cl_mem inA_mem = get_image_from_tensor(inputs[0], ocl_backend_, temp_tensor_storage);
        cl_mem out_mem_handle = ocl_backend_->get_cl_mem(*output);

        clSetKernelArg(kernel_to_use, 0, sizeof(cl_sampler), &sampler_);
        clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &inA_mem);
        if (input_dtype == MLLM_TYPE_F32) {
            clSetKernelArg(kernel_to_use, 2, sizeof(float), &neg_data);
        } else {
            mllm_fp16_t data_fp16 = MLLM_FP32_TO_FP16(neg_data);
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
        // Buffer 路径
        cl_kernel kernel_to_use = (input_dtype == MLLM_TYPE_F32) ? kernel_fp32_buffer_ : kernel_fp16_buffer_;
        cl_mem in0_buf = ocl_backend_->get_cl_mem(*inputs[0]);
        cl_mem out_buf = ocl_backend_->get_cl_mem(*output);

        clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &in0_buf);
        if (input_dtype == MLLM_TYPE_F32) {
            clSetKernelArg(kernel_to_use, 1, sizeof(float), &neg_data);
        } else {
            mllm_fp16_t data_fp16 = MLLM_FP32_TO_FP16(neg_data);
            clSetKernelArg(kernel_to_use, 1, sizeof(mllm_fp16_t), &data_fp16);
        }
        clSetKernelArg(kernel_to_use, 2, sizeof(cl_mem), &out_buf);

        size_t count = inputs[0]->count();
        if (input_dtype == MLLM_TYPE_F16) { count /= 4; }

        const size_t global_work_size[1] = {count};
        cl_event event;
        clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 1, nullptr, global_work_size, nullptr, 0, nullptr, &event);
        ocl_backend_->addProfilingEvent(this->name() + "sub", event);
    }

    return MLLM_NO_ERROR;
}

} // namespace mllm