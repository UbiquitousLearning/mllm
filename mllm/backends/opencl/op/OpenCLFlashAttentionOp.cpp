#include "OpenCLFlashAttentionOp.hpp"
#include "Types.hpp"
#include "utils/OpenCLTools.hpp"

// 宏定义现在只控制Tile的行列数和工作组大小
#define Br 8
#define Bc 8
#define WGS_S 8
#define WGS_D 8
// 定义一个与内核中 LOCAL_MEM_SIZE 计算逻辑相关的维度上限
#define DIM_MAX 128
// 定义工作组大小，必须与内核中的WGS一致
#define WGS 128

namespace mllm {

OpenCLFlashAttentionOp::OpenCLFlashAttentionOp(Backend *bn, std::string name, bool causal_mask) :
    Op(bn, std::move(name)), causal_mask_(causal_mask) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    const std::string kernel_path = "kernel/flash_attention.cl";
    std::string build_options;
    if (ocl_backend_->has_fp16_support()) {
        build_options += " -DSUPPORTS_FP16";
    }
    cl_program program = ocl_backend_->getProgram(kernel_path, build_options);

    cl_int err;
    kernel_fp32_ = clCreateKernel(program, "flash_attention_2_prefill_fp32", &err);
    check_cl_error(err, "clCreateKernel for flash_attention_2_prefill_fp32");

    kernel_fp32_decode_ = clCreateKernel(program, "flash_attention_2_decode_fp32", &err);
    check_cl_error(err, "clCreateKernel for flash_attention_2_decode_fp32");

    kernel_fp16_ = clCreateKernel(program, "flash_attention_2_prefill_fp16", &err);
    check_cl_error(err, "clCreateKernel for flash_attention_2_prefill_fp16");

    kernel_fp16_decode_ = clCreateKernel(program, "flash_attention_2_decode_fp16", &err);
    check_cl_error(err, "clCreateKernel for flash_attention_2_decode_fp16");

    kernel_fp32_image_ = clCreateKernel(program, "flash_attention_2_prefill_fp32_image", &err);
    check_cl_error(err, "clCreateKernel for flash_attention_2_prefill_fp32_image");
    kernel_fp32_decode_image_ = clCreateKernel(program, "flash_attention_2_decode_fp32_image", &err);
    check_cl_error(err, "clCreateKernel for flash_attention_2_decode_fp32_image");
    kernel_fp16_image_ = clCreateKernel(program, "flash_attention_2_prefill_fp16_image", &err);
    check_cl_error(err, "clCreateKernel for flash_attention_2_prefill_fp16_image");
    kernel_fp16_decode_image_ = clCreateKernel(program, "flash_attention_2_decode_fp16_image", &err);
    check_cl_error(err, "clCreateKernel for flash_attention_2_decode_fp16_image");

    sampler_ = clCreateSampler(ocl_backend_->getContext(), CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    check_cl_error(err, "clCreateSampler for FlashAttention");
}

OpenCLFlashAttentionOp::~OpenCLFlashAttentionOp() {
    if (kernel_fp32_) clReleaseKernel(kernel_fp32_);
    if (kernel_fp32_decode_) clReleaseKernel(kernel_fp32_decode_);
    if (kernel_fp16_) clReleaseKernel(kernel_fp16_);
    if (kernel_fp16_decode_) clReleaseKernel(kernel_fp16_decode_);
    if (kernel_fp32_image_) clReleaseKernel(kernel_fp32_image_);
    if (kernel_fp32_decode_image_) clReleaseKernel(kernel_fp32_decode_image_);
    if (kernel_fp16_image_) clReleaseKernel(kernel_fp16_image_);
    if (kernel_fp16_decode_image_) clReleaseKernel(kernel_fp16_decode_image_);
    if (sampler_) clReleaseSampler(sampler_);
}

ErrorCode OpenCLFlashAttentionOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto q_tensor = inputs[0];
    auto o_tensor = outputs[0];
    o_tensor->reshape(q_tensor->batch(), q_tensor->head(), q_tensor->sequence(), q_tensor->dimension());
    o_tensor->setDtype(q_tensor->dtype());
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLFlashAttentionOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    for (auto &input : inputs) {
        input->to(MLLM_OPENCL);
    }
    outputs[0]->setDtype(inputs[0]->dtype());
    outputs[0]->to(MLLM_OPENCL);
    outputs[0]->alloc();
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLFlashAttentionOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto q_tensor = inputs[0];
    auto k_tensor = inputs[1];
    auto v_tensor = inputs[2];
    auto o_tensor = outputs[0];

    const auto data_type = q_tensor->dtype();

    const int dim_size = q_tensor->dimension();

    if (dim_size > DIM_MAX) {
        throw std::runtime_error("FlashAttention Error: Tensor dimension size (" + std::to_string(dim_size) + ") exceeds kernel's compiled limit (DIM_MAX=" + std::to_string(DIM_MAX) + ").");
    }

    cl_mem q_buf = ocl_backend_->get_cl_mem(*q_tensor);
    cl_mem k_buf = ocl_backend_->get_cl_mem(*k_tensor);
    cl_mem v_buf = ocl_backend_->get_cl_mem(*v_tensor);
    cl_mem o_buf = ocl_backend_->get_cl_mem(*o_tensor);

    const int batch_size = q_tensor->batch();
    const int q_head_size = q_tensor->head();
    const int kv_head_size = k_tensor->head();
    const int seq_size_q = q_tensor->sequence();
    const int seq_size_k = k_tensor->sequence();

    int causal_mask_int = causal_mask_ ? 1 : 0;
    if (seq_size_q == 1) {
        causal_mask_int = 0;
    }
    // 2. 决策：是否使用全 Image 优化路径

#if !defined(__APPLE__) || !defined(__aarch64__)
    bool use_image_path = (seq_size_q > 1) && (q_tensor->dimension() % 4 == 0) && (k_tensor->dimension() % 4 == 0) && (v_tensor->dimension() % 4 == 0) && (o_tensor->dimension() % 4 == 0);
#else
    bool use_image_path = false;
#endif

    // bool use_image_path = false;
    if (use_image_path) {
        // a. 将所有相关张量原地转换为 Image
        tensorGlobal2Image(*q_tensor);
        tensorGlobal2Image(*k_tensor);
        tensorGlobal2Image(*v_tensor);
        tensorGlobal2Image(*o_tensor);
        cl_mem q_img = ocl_backend_->get_cl_mem(*q_tensor);
        cl_mem k_img = ocl_backend_->get_cl_mem(*k_tensor);
        cl_mem v_img = ocl_backend_->get_cl_mem(*v_tensor);
        cl_mem o_img = ocl_backend_->get_cl_mem(*o_tensor);
        cl_kernel kernel_to_use = nullptr;
        cl_event event;
        cl_int err;
        if (seq_size_q == 1) { // Decode 阶段 (GEMV-like)
            if (data_type == MLLM_TYPE_F32) {
                kernel_to_use = kernel_fp32_decode_image_;
            } else { // MLLM_TYPE_F16
                kernel_to_use = kernel_fp16_decode_image_;
            }
            int arg_idx = 0;
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_sampler), &sampler_);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &q_img);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &k_img);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &v_img);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &o_img);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &q_head_size);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &kv_head_size);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &seq_size_k);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &dim_size);
            const size_t global_work_size[2] = {
                o_tensor->device_memory().image_width, // N / 4
                o_tensor->device_memory().image_height // B * H
            };
            err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 2, nullptr, global_work_size, nullptr, 0, nullptr, &event);
            ocl_backend_->addProfilingEvent(this->name() + "flash_attention2_decode_image", event);
            check_cl_error(err, "clEnqueueNDRangeKernel for FlashAttention Decode Image");
        } else { // Prefill 阶段 (GEMM-like)
            if (data_type == MLLM_TYPE_F32) {
                kernel_to_use = kernel_fp32_image_;
            } else { // MLLM_TYPE_F16
                kernel_to_use = kernel_fp16_image_;
            }
            int arg_idx = 0;
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_sampler), &sampler_);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &q_img);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &k_img);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &v_img);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &o_img);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &q_head_size);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &kv_head_size);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &seq_size_q);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &seq_size_k);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &dim_size);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &causal_mask_int);
            const size_t local_work_size[1] = {WGS};
            const size_t num_output_rows = batch_size * q_head_size * seq_size_q;
            const size_t global_work_size[1] = {num_output_rows * WGS};
            err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 1, nullptr, global_work_size, local_work_size, 0, nullptr, &event);
            ocl_backend_->addProfilingEvent(this->name() + "flash_attention2_prefill_image", event);
            check_cl_error(err, "clEnqueueNDRangeKernel for FlashAttention Prefill Image");
        }

        // d. 将所有张量转换回 Buffer 以便后续操作
        tensorImage2Global(*q_tensor);
        tensorImage2Global(*k_tensor);
        tensorImage2Global(*v_tensor);
        tensorImage2Global(*o_tensor);
    } else {
        if (data_type == MLLM_TYPE_F32 && seq_size_q == 1) { // Decode 阶段 - FP32
            cl_kernel kernel = kernel_fp32_decode_;
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &q_buf);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &k_buf);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &v_buf);
            clSetKernelArg(kernel, 3, sizeof(cl_mem), &o_buf);
            clSetKernelArg(kernel, 4, sizeof(int), &q_head_size);
            clSetKernelArg(kernel, 5, sizeof(int), &kv_head_size);
            clSetKernelArg(kernel, 6, sizeof(int), &seq_size_k);
            clSetKernelArg(kernel, 7, sizeof(int), &dim_size);
            const size_t local_work_size[3] = {WGS, 1, 1};
            const size_t num_output_rows = batch_size * q_head_size;
            size_t global_work_size[3] = {num_output_rows * WGS, 1, 1};
            cl_event event;
            cl_int err = clEnqueueNDRangeKernel(
                ocl_backend_->getQueue(), kernel, 1, nullptr,
                global_work_size, local_work_size, 0, nullptr, &event);
            ocl_backend_->addProfilingEvent(this->name() + "flash_attention2", event);
            check_cl_error(err, "clEnqueueNDRangeKernel for FlashAttention Decode");
        } else if (data_type == MLLM_TYPE_F32) { // Prefill 阶段 - FP32
            cl_kernel kernel = kernel_fp32_;
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &q_buf);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &k_buf);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &v_buf);
            clSetKernelArg(kernel, 3, sizeof(cl_mem), &o_buf);
            clSetKernelArg(kernel, 4, sizeof(int), &q_head_size);
            clSetKernelArg(kernel, 5, sizeof(int), &kv_head_size);
            clSetKernelArg(kernel, 6, sizeof(int), &seq_size_q);
            clSetKernelArg(kernel, 7, sizeof(int), &seq_size_k);
            clSetKernelArg(kernel, 8, sizeof(int), &dim_size);
            clSetKernelArg(kernel, 9, sizeof(int), &causal_mask_int);
            const size_t local_work_size[3] = {WGS, 1, 1};
            const size_t num_output_rows = batch_size * q_head_size * seq_size_q;
            size_t global_work_size[3] = {num_output_rows * WGS, 1, 1};
            cl_event event;
            cl_int err = clEnqueueNDRangeKernel(
                ocl_backend_->getQueue(), kernel, 1, nullptr,
                global_work_size, local_work_size, 0, nullptr, &event);
            ocl_backend_->addProfilingEvent(this->name() + "flash_attention2", event);
            check_cl_error(err, "clEnqueueNDRangeKernel for FlashAttention V2");
        } else if (data_type == MLLM_TYPE_F16 && seq_size_q == 1) { // Decode 阶段 - FP16
            cl_kernel kernel = kernel_fp16_decode_;
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &q_buf);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &k_buf);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &v_buf);
            clSetKernelArg(kernel, 3, sizeof(cl_mem), &o_buf);
            clSetKernelArg(kernel, 4, sizeof(int), &q_head_size);
            clSetKernelArg(kernel, 5, sizeof(int), &kv_head_size);
            clSetKernelArg(kernel, 6, sizeof(int), &seq_size_k);
            clSetKernelArg(kernel, 7, sizeof(int), &dim_size);
            const size_t local_work_size[3] = {WGS, 1, 1};
            const size_t num_output_rows = batch_size * q_head_size;
            size_t global_work_size[3] = {num_output_rows * WGS, 1, 1};
            cl_event event;
            cl_int err = clEnqueueNDRangeKernel(
                ocl_backend_->getQueue(), kernel, 1, nullptr,
                global_work_size, local_work_size, 0, nullptr, &event);
            ocl_backend_->addProfilingEvent(this->name() + "flash_attention2_fp16_decode", event);
            check_cl_error(err, "clEnqueueNDRangeKernel for FlashAttention Decode FP16");
        } else if (data_type == MLLM_TYPE_F16) { // Prefill 阶段 - FP16
            cl_kernel kernel = kernel_fp16_;
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &q_buf);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &k_buf);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &v_buf);
            clSetKernelArg(kernel, 3, sizeof(cl_mem), &o_buf);
            clSetKernelArg(kernel, 4, sizeof(int), &q_head_size);
            clSetKernelArg(kernel, 5, sizeof(int), &kv_head_size);
            clSetKernelArg(kernel, 6, sizeof(int), &seq_size_q);
            clSetKernelArg(kernel, 7, sizeof(int), &seq_size_k);
            clSetKernelArg(kernel, 8, sizeof(int), &dim_size);
            clSetKernelArg(kernel, 9, sizeof(int), &causal_mask_int);
            const size_t local_work_size[3] = {WGS, 1, 1};
            const size_t num_output_rows = batch_size * q_head_size * seq_size_q;
            size_t global_work_size[3] = {num_output_rows * WGS, 1, 1};
            cl_event event;
            cl_int err = clEnqueueNDRangeKernel(
                ocl_backend_->getQueue(), kernel, 1, nullptr,
                global_work_size, local_work_size, 0, nullptr, &event);
            ocl_backend_->addProfilingEvent(this->name() + "flash_attention2_fp16", event);
            check_cl_error(err, "clEnqueueNDRangeKernel for FlashAttention V2 FP16");

        } else {
            return NOT_SUPPORT;
        }
    }

    return MLLM_NO_ERROR;
}
} // namespace mllm