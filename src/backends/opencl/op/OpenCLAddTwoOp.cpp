#include "OpenCLAddTwoOp.hpp"
#include "Types.hpp"
#include "utils/OpenCLTools.hpp"

namespace mllm {

OpenCLAddTwoOp::OpenCLAddTwoOp(Backend *bn, std::string name) :
    Op(bn, std::move(name)) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    // 只获取一次 Program
    const std::string kernel_path = "kernel/add.cl";
    cl_program program = ocl_backend_->getProgram(kernel_path);

    cl_int err;
    // --- 创建全部四个内核 ---
    kernel_fp32_buffer_ = clCreateKernel(program, "add_float", &err);
    check_cl_error(err, "clCreateKernel for add_float");

    kernel_fp32_image_ = clCreateKernel(program, "add_float_image2d", &err);
    check_cl_error(err, "clCreateKernel for add_float_image2d");

    kernel_fp16_buffer_ = clCreateKernel(program, "add_fp16_vector", &err);
    check_cl_error(err, "clCreateKernel for add_fp16_vector");

    kernel_fp16_image_ = clCreateKernel(program, "add_fp16_image2d", &err);
    check_cl_error(err, "clCreateKernel for add_fp16_image2d");

    // --- 创建 Sampler ---
    sampler_ = clCreateSampler(ocl_backend_->getContext(), CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    check_cl_error(err, "clCreateSampler");
}

// 替换您的析构函数
OpenCLAddTwoOp::~OpenCLAddTwoOp() {
    if (kernel_fp32_buffer_) clReleaseKernel(kernel_fp32_buffer_);
    if (kernel_fp32_image_) clReleaseKernel(kernel_fp32_image_);
    if (kernel_fp16_buffer_) clReleaseKernel(kernel_fp16_buffer_);
    if (kernel_fp16_image_) clReleaseKernel(kernel_fp16_image_);
    if (sampler_) clReleaseSampler(sampler_);
}

ErrorCode OpenCLAddTwoOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // 加法操作要求输入和输出的形状一致
    auto input0_shape = inputs[0]->shape();
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    outputs[0]->setDtype(inputs[0]->dtype());
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLAddTwoOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // 确保输入在设备上
    for (auto &input : inputs) {
        input->to(MLLM_OPENCL);
    }
    auto output = outputs[0];
    output->to(MLLM_OPENCL);
    // 根据输入设置输出的数据类型
    output->setDtype(inputs[0]->dtype());

    auto &out_mem = output->device_memory();

    // **核心修改：直接决策为 Image 或 Buffer**
    if (output->dimension() % 4 == 0 && false) {
        // 条件满足，直接为输出张量申请 Image2D 类型的内存
        out_mem.type = MEM_TYPE_IMAGE_2D; // **直接设为 Image2D**
        out_mem.image_width = output->dimension() / 4;
        out_mem.image_height = output->batch() * output->head() * output->sequence();
    } else {
        // 条件不满足，回退到普通 Buffer
        out_mem.type = MEM_TYPE_BUFFER;
    }

    // alloc() 现在会根据 out_mem.type 直接创建出 Image 或 Buffer
    output->alloc();
    return MLLM_NO_ERROR;
}
ErrorCode OpenCLAddTwoOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input1 = inputs[0];
    auto input2 = inputs[1];
    auto output = outputs[0];

    // 决定是否走Image优化路径
    bool use_image_path = (output->dimension() % 4 == 0) && false;

    if (use_image_path) {
        // ===================================================
        // ================ Image 优化路径 ===================
        // ===================================================

        // 1. 将两个输入和输出Tensor原地转换为Image2D类型
        tensorGlobal2Image(*input1);
        tensorGlobal2Image(*input2);
        tensorGlobal2Image(*output);

        // 2. 选择内核
        cl_kernel kernel_to_use = (input1->dtype() == MLLM_TYPE_F32) ? kernel_fp32_image_ : kernel_fp16_image_;

        // 3. 获取Image句柄并设置参数
        cl_mem inA_img = ocl_backend_->get_cl_mem(*input1);
        cl_mem inB_img = ocl_backend_->get_cl_mem(*input2);
        cl_mem out_img = ocl_backend_->get_cl_mem(*output);

        clSetKernelArg(kernel_to_use, 0, sizeof(cl_sampler), &sampler_);
        clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &inA_img);
        clSetKernelArg(kernel_to_use, 2, sizeof(cl_mem), &inB_img);
        clSetKernelArg(kernel_to_use, 3, sizeof(cl_mem), &out_img);

        const int width = static_cast<int>(output->device_memory().image_width);
        const int height = static_cast<int>(output->device_memory().image_height);
        clSetKernelArg(kernel_to_use, 4, sizeof(int), &width);
        clSetKernelArg(kernel_to_use, 5, sizeof(int), &height);

        // 4. 执行内核
        const size_t global_work_size[2] = {(size_t)width, (size_t)height};
        cl_event event;
        clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 2, nullptr, global_work_size, nullptr, 0, nullptr, &event);
        ocl_backend_->addProfilingEvent(this->name() + "add2", event);

        // 5. [重要] 将所有Tensor转换回Buffer格式，以供后续算子使用
        tensorImage2Global(*input1);
        tensorImage2Global(*input2);
        tensorImage2Global(*output);

    } else {
        // ===================================================
        // ============= 普通 Buffer 回退路径 ================
        // ===================================================

        // 确保所有Tensor都是Buffer格式
        tensorImage2Global(*input1);
        tensorImage2Global(*input2);
        tensorImage2Global(*output);

        cl_kernel kernel_to_use = (input1->dtype() == MLLM_TYPE_F32) ? kernel_fp32_buffer_ : kernel_fp16_buffer_;

        cl_mem in0_buf = ocl_backend_->get_cl_mem(*input1);
        cl_mem in1_buf = ocl_backend_->get_cl_mem(*input2);
        cl_mem out_buf = ocl_backend_->get_cl_mem(*output);

        clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &in0_buf);
        clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &in1_buf);
        clSetKernelArg(kernel_to_use, 2, sizeof(cl_mem), &out_buf);

        size_t count = input1->count();
        if (input1->dtype() == MLLM_TYPE_F16) {
            if (count % 4 != 0) {
                throw std::runtime_error("[addTwo]For FP16 vector kernel, tensor count must be a multiple of 4.");
            }
            count /= 4;
        }

        const size_t global_work_size[1] = {count};
        cl_event event;
        clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 1, nullptr, global_work_size, nullptr, 0, nullptr, &event);
        ocl_backend_->addProfilingEvent(this->name() + "add2", event);
    }

    return MLLM_NO_ERROR;
}
} // namespace mllm