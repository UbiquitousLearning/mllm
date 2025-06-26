#include "OpenCLAddFuncOp.hpp"
#include "Types.hpp"
#include "utils/OpenCLTools.hpp"

namespace mllm {


OpenCLAddFuncOp::OpenCLAddFuncOp(Backend *bn, std::string name) : Op(bn, std::move(name)) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    // 只获取一次 Program
    const std::string kernel_path_str = get_kernel_path(__FILE__, "../kernel/add.cl");
    cl_program program = ocl_backend_->getProgram(kernel_path_str);

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
OpenCLAddFuncOp::~OpenCLAddFuncOp() {
    if (kernel_fp32_buffer_) clReleaseKernel(kernel_fp32_buffer_);
    if (kernel_fp32_image_) clReleaseKernel(kernel_fp32_image_);
    if (kernel_fp16_buffer_) clReleaseKernel(kernel_fp16_buffer_);
    if (kernel_fp16_image_) clReleaseKernel(kernel_fp16_image_);
    if (sampler_) clReleaseSampler(sampler_);
}

ErrorCode OpenCLAddFuncOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // 加法操作要求输入和输出的形状一致
    auto input0_shape = inputs[0]->shape();
    outputs[0]->reshape(input0_shape[0], input0_shape[1], input0_shape[2], input0_shape[3]);
    outputs[0]->setDtype(inputs[0]->dtype());
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLAddFuncOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // 确保输入在设备上
    for(auto& input : inputs) {
        input->to(MLLM_OPENCL);
    }
    auto output = outputs[0];

    // 根据输入设置输出的数据类型
    output->setDtype(inputs[0]->dtype());

    auto& out_mem = output->device_memory();
    
    // **核心修改：直接决策为 Image 或 Buffer**
    if (output->dimension() % 4 == 0) {
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

// in OpenCLAddFuncOp.cpp

ErrorCode OpenCLAddFuncOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // 1. 获取输入数据类型和输出张量
    auto input_dtype = inputs[0]->dtype();
    auto output = outputs[0];

    // 2. 根据输出张量的内存类型，决定执行“Image优化路径”还是“普通Buffer路径”
    //    这个类型是在 setUp 函数中根据维度是否为4的倍数决定的。
    if (output->device_memory().type == MEM_TYPE_IMAGE_2D) {
        // ===================================================
        // ================ Image 优化路径 ===================
        // ===================================================

        // a. 选择内核：根据数据类型选择FP32或FP16的Image内核
        cl_kernel kernel_to_use = nullptr;
        if (input_dtype == MLLM_TYPE_F32) {
            kernel_to_use = kernel_fp32_image_;
        } else if (input_dtype == MLLM_TYPE_F16) {
            kernel_to_use = kernel_fp16_image_;
        } else {
            throw std::runtime_error("Unsupported data type for OpenCLAddFuncOp Image Path");
        }

        // b. 准备内核参数
        //    - 输入张量(inputs): 它们是由默认的 .cl() 方法创建的Buffer，所以必须通过工具函数进行拷贝转换。
        //    - 输出张量(output): 它在setUp中已经被直接创建为Image，所以可以直接获取其句柄。
        
        //    创建一个临时存储vector，用于管理为“输入”转换而来的临时Image的生命周期
        std::vector<Tensor> temp_tensor_storage;
        
        cl_mem inA_mem = get_image_from_tensor(inputs[0], ocl_backend_, temp_tensor_storage);
        cl_mem inB_mem = get_image_from_tensor(inputs[1], ocl_backend_, temp_tensor_storage);
        
        // 输出张量已经是Image，直接获取句柄，无需转换！
        cl_mem out_mem_handle = ocl_backend_->get_cl_mem(*output);

        // c. 设置内核参数
        clSetKernelArg(kernel_to_use, 0, sizeof(cl_sampler), &sampler_);
        clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &inA_mem);
        clSetKernelArg(kernel_to_use, 2, sizeof(cl_mem), &inB_mem);
        clSetKernelArg(kernel_to_use, 3, sizeof(cl_mem), &out_mem_handle);

        const int width = static_cast<int>(output->device_memory().image_width);
        const int height = static_cast<int>(output->device_memory().image_height);

        clSetKernelArg(kernel_to_use, 4, sizeof(int), &width);
        clSetKernelArg(kernel_to_use, 5, sizeof(int), &height);

        // d. 设置工作维度并执行内核
        const size_t global_work_size[2] = { (size_t)width, (size_t)height };
        clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);

    } else {
        // ===================================================
        // ============= 普通 Buffer 回退路径 ================
        // ===================================================

        // a. 选择内核：根据数据类型选择FP32或FP16的Buffer内核
        cl_kernel kernel_to_use = nullptr;
        if (input_dtype == MLLM_TYPE_F32) {
            kernel_to_use = kernel_fp32_buffer_;
        } else if (input_dtype == MLLM_TYPE_F16) {
            kernel_to_use = kernel_fp16_buffer_;
        } else {
            throw std::runtime_error("Unsupported data type for OpenCLAddFuncOp Buffer Path");
        }
        
        // b. 准备内核参数 (所有张量都是Buffer，直接获取句柄)
        cl_mem in0_buf = ocl_backend_->get_cl_mem(*inputs[0]);
        cl_mem in1_buf = ocl_backend_->get_cl_mem(*inputs[1]);
        cl_mem out_buf = ocl_backend_->get_cl_mem(*output);

        clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &in0_buf);
        clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &in1_buf);
        clSetKernelArg(kernel_to_use, 2, sizeof(cl_mem), &out_buf);

        // c. 计算工作项数量
        size_t count = inputs[0]->count();
        // FP16的Buffer内核是向量化的，工作项数量是总元素数除以4
        if (input_dtype == MLLM_TYPE_F16) {
            if (count % 4 != 0) {
                 throw std::runtime_error("For FP16 vector kernel, tensor count must be a multiple of 4.");
            }
            count /= 4;
        }
        
        // d. 设置工作维度并执行内核
        const size_t global_work_size[1] = {count};
        clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    }

    return MLLM_NO_ERROR;
}
} // namespace mllm