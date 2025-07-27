#include "OpenCLLinearOp.hpp"
#include "Backend.hpp"
#include "DataType.hpp"
#include "Types.hpp"
#include "utils/OpenCLTools.hpp"
#include <cassert>
// #include <iostream>
#include <iostream>
#include <string>

namespace mllm {

OpenCLLinearOp::OpenCLLinearOp(Backend *bn, string opName, int in_features, int out_features, bool bias) :
    Op(bn, opName), in_features_(in_features), out_features_(out_features), support_bias_(bias) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) {
        throw std::runtime_error("Backend for OpenCLLinearOp is not OpenCLBackend");
    }

    const std::string kernel_path = "kernel/matmul_transb_bias.cl";
    std::string build_options;
    if (ocl_backend_->has_fp16_support()) {
        build_options += " -DSUPPORTS_FP16";
    }

    cl_program program = ocl_backend_->getProgram(kernel_path, build_options);
    cl_int err;
    kernel_fp32_transb_bias_ = clCreateKernel(program, "gemm_fp32_transb_bias", &err);
    check_cl_error(err, "CreateKernel gemm_fp32_transb_bias");
    kernel_fp16_transb_bias_ = clCreateKernel(program, "gemm_fp16_transb_bias", &err);
    check_cl_error(err, "CreateKernel gemm_fp16_transb_bias");
    kernel_fp16_q4_0_transb_bias_ = clCreateKernel(program, "gemm_fp16_q4_0_transb_bias", &err);
    check_cl_error(err, "CreateKernel gemm_fp16_q4_0_transb_bias");
    kernel_fp32_q4_0_transb_bias_ = clCreateKernel(program, "gemm_fp32_q4_0_transb_bias", &err);
    check_cl_error(err, "CreateKernel gemm_fp32_q4_0_transb_bias");
    kernel_gemv_fp32_q4_0_transb_bias_ = clCreateKernel(program, "gemv_fp32_q4_0_transb_bias", &err);
    check_cl_error(err, "CreateKernel gemv_fp32_q4_0_transb_bias");
    kernel_gemv_fp16_q4_0_transb_bias_ = clCreateKernel(program, "gemv_fp16_q4_0_transb_bias", &err);
    check_cl_error(err, "CreateKernel gemv_fp16_q4_0_transb_bias");
    if (ocl_backend_->has_fp16_support()) {
        kernel_gemv_fp16_q4_0_transb_bias_half16_ = clCreateKernel(program, "gemv_fp16_q4_0_transb_bias_half16", &err);
        check_cl_error(err, "CreateKernel gemv_fp16_q4_0_transb_bias_half16");
    }

    kernel_fp32_q4_0_transb_bias_image2d_ = clCreateKernel(program, "gemm_fp32_q4_0_transb_bias_image_pipe", &err);
    check_cl_error(err, "CreateKernel gemm_fp32_q4_0_transb_bias_image_pipe");
    kernel_fp16_q4_0_transb_bias_image2d_ = clCreateKernel(program, "gemm_fp16_q4_0_transb_bias_image_pipe", &err);
    check_cl_error(err, "CreateKernel gemm_fp16_q4_0_transb_bias_image_pipe");
    kernel_gemv_fp32_q4_0_transb_bias_image2d_ = clCreateKernel(program, "gemv_fp32_q4_0_transb_bias_image_pipe", &err);
    check_cl_error(err, "CreateKernel gemv_fp32_q4_0_transb_bias_image_pipe");
    kernel_gemv_fp16_q4_0_transb_bias_image2d_ = clCreateKernel(program, "gemv_fp16_q4_0_transb_bias_image_pipe", &err);
    check_cl_error(err, "CreateKernel gemv_fp16_q4_0_transb_bias_image_pipe");
}

OpenCLLinearOp::~OpenCLLinearOp() {
    if (kernel_fp32_transb_bias_) clReleaseKernel(kernel_fp32_transb_bias_);
    if (kernel_fp16_transb_bias_) clReleaseKernel(kernel_fp16_transb_bias_);
    if (kernel_fp32_q4_0_transb_bias_) clReleaseKernel(kernel_fp32_q4_0_transb_bias_);
    if (kernel_fp16_q4_0_transb_bias_) clReleaseKernel(kernel_fp16_q4_0_transb_bias_);
    if (kernel_gemv_fp32_q4_0_transb_bias_) clReleaseKernel(kernel_gemv_fp32_q4_0_transb_bias_);
    if (kernel_gemv_fp16_q4_0_transb_bias_) clReleaseKernel(kernel_gemv_fp16_q4_0_transb_bias_);
    if (kernel_gemv_fp16_q4_0_transb_bias_half16_) clReleaseKernel(kernel_gemv_fp16_q4_0_transb_bias_half16_);
    if (kernel_fp32_q4_0_transb_bias_image2d_) clReleaseKernel(kernel_fp32_q4_0_transb_bias_image2d_);
    if (kernel_fp16_q4_0_transb_bias_image2d_) clReleaseKernel(kernel_fp16_q4_0_transb_bias_image2d_);
    if (kernel_gemv_fp32_q4_0_transb_bias_image2d_) clReleaseKernel(kernel_gemv_fp32_q4_0_transb_bias_image2d_);
    if (kernel_gemv_fp16_q4_0_transb_bias_image2d_) clReleaseKernel(kernel_gemv_fp16_q4_0_transb_bias_image2d_);
}

ErrorCode OpenCLLinearOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);

    // Input:  [batch, 1, seq_len, in_features]
    // Output: [batch, 1, seq_len, out_features]
    assert(inputs[0]->head() == 1);
    assert(in_features_ == inputs[0]->dimension());

    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), out_features_);
    outputs[0]->setDtype(inputs[0]->dtype());
    // #if !defined(__APPLE__) || !defined(__aarch64__)
    // const size_t max_image_width = ocl_backend_->getMaxImage2dWidth();
    // if (out_features_ % 4 == 0 && (out_features_ / 4) <= max_image_width) { // inputs[0]->sequence() == 1 &&
    //     auto &out_mem = outputs[0]->device_memory();
    //     out_mem.type = MEM_TYPE_IMAGE_2D;
    //     out_mem.image_width = out_features_ / 4;
    //     out_mem.image_height = inputs[0]->batch() * inputs[0]->head() * inputs[0]->sequence();
    // }
    // #endif
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLLinearOp::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.setBackend(ocl_backend_);
    weight_.reshape(1, 1, out_features_, in_features_);
    weight_.setDtype(loader.getDataType(weight_.name()));
    weight_.alloc();
    loader.load(&weight_);
    // weight_.saveQ4Data_d();

    if (support_bias_) {
        bias_.setName(name() + ".bias");
        bias_.setBackend(ocl_backend_);
        bias_.reshape(1, 1, 1, out_features_);
        bias_.setDtype(loader.getDataType(bias_.name()));
        bias_.alloc();
        loader.load(&bias_);
    }
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLLinearOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // Move all tensors to OpenCL device
    inputs[0]->to(MLLM_OPENCL);
    outputs[0]->to(MLLM_OPENCL);
    // Allocate output memory
    outputs[0]->alloc();
    return MLLM_NO_ERROR;
}
ErrorCode OpenCLLinearOp::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.unload();
    if (support_bias_) {
        bias_.unload();
    }
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLLinearOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto &A = inputs[0];
    auto &W = weight_;
    auto &C = outputs[0];

    // 1. 选择 Matmul+Bias 内核
    cl_kernel kernel_to_use = nullptr;
    // 2. 设置参数并执行
    const int M = C->sequence();
    const int K = A->dimension();
    const int N = C->dimension();
    const int B_size = A->batch();
    const int H_size = A->head();
    const int K_b = W.dimension();

    cl_mem a_mem = ocl_backend_->get_cl_mem(*A);
    cl_mem w_mem = ocl_backend_->get_cl_mem(W);
    cl_mem c_mem = ocl_backend_->get_cl_mem(*C);

    cl_mem bias_mem_arg = support_bias_ ? ocl_backend_->get_cl_mem(bias_) : a_mem; // 使用 a_mem 作为哑参数
    const int has_bias_flag = support_bias_ ? 1 : 0;
    cl_event event;
    cl_int err;

    if (M == 1 && (A->dtype() == MLLM_TYPE_F32 || A->dtype() == MLLM_TYPE_F16) && W.dtype() == MLLM_TYPE_Q4_0) {
        bool use_image_path_for_gemv = (A->dimension() % 4 == 0 && C->dimension() % 4 == 0) && (C->device_memory().type == MEM_TYPE_IMAGE_2D);
        if (use_image_path_for_gemv) {
            // --- GEMV All Image 路径 ---
            tensorGlobal2Image(*A);
            tensorGlobal2Image(*C);
            cl_kernel kernel_to_use = nullptr;
            if (A->dtype() == MLLM_TYPE_F32) {
                kernel_to_use = kernel_gemv_fp32_q4_0_transb_bias_image2d_;
            } else { // MLLM_TYPE_F16
                kernel_to_use = kernel_gemv_fp16_q4_0_transb_bias_image2d_;
            }
            cl_mem a_img_mem = ocl_backend_->get_cl_mem(*A);
            cl_mem c_img_mem = ocl_backend_->get_cl_mem(*C);
            cl_sampler sampler = clCreateSampler(ocl_backend_->getContext(), CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
            check_cl_error(err, "clCreateSampler for GEMV Image");
            int arg_idx = 0;
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_sampler), &sampler);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &a_img_mem);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &w_mem);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &bias_mem_arg);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &c_img_mem);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &K);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &N);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &H_size);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &has_bias_flag);
            const size_t global_work_size[2] = {(size_t)N / 4, 1}; // Width: N/4, Height: 1
            err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 1, nullptr, global_work_size, nullptr, 0, nullptr, &event);
            ocl_backend_->addProfilingEvent(this->name() + "_gemv_image", event);
            check_cl_error(err, "EnqueueNDRangeKernel GEMV Image");
            clReleaseSampler(sampler);
            tensorImage2Global(*A);
            tensorImage2Global(*C);
        } else {
            cl_kernel kernel_to_use = nullptr;
            if (A->dtype() == MLLM_TYPE_F32) {
                kernel_to_use = kernel_gemv_fp32_q4_0_transb_bias_;
            } else { // MLLM_TYPE_F16
                kernel_to_use = kernel_gemv_fp16_q4_0_transb_bias_;
            }
            int arg_idx = 0;
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &a_mem);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &w_mem);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &bias_mem_arg);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &c_mem);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &K);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &N);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &H_size);
            clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &has_bias_flag);
            if (ocl_backend_->has_fp16_support() && A->dtype() == MLLM_TYPE_F16) {
                const size_t local_work_size[2] = {128, 1}; // 建议从128开始
                const size_t global_work_size[2] = {(size_t)N * local_work_size[0], (size_t)(B_size * H_size)};
                err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 2, nullptr, global_work_size, local_work_size, 0, nullptr, &event);
            } else {
                const size_t local_work_size[2] = {256, 1};
                const size_t global_work_size[2] = {(size_t)N * local_work_size[0], (size_t)(B_size * H_size)};
                err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 2, nullptr, global_work_size, local_work_size, 0, nullptr, &event);
            }
            ocl_backend_->addProfilingEvent(this->name(), event);
            check_cl_error(err, "EnqueueNDRangeKernel GEMV");
        }
    } else if (C->device_memory().type == MEM_TYPE_IMAGE_2D) {
        tensorGlobal2Image(*inputs[0]);
        tensorGlobal2Image(*C);
        cl_kernel kernel_to_use = nullptr;
        if (A->dtype() == MLLM_TYPE_F32 && W.dtype() == MLLM_TYPE_Q4_0) {
            kernel_to_use = kernel_fp32_q4_0_transb_bias_image2d_;
        } else if (A->dtype() == MLLM_TYPE_F16 && W.dtype() == MLLM_TYPE_Q4_0) {
            kernel_to_use = kernel_fp16_q4_0_transb_bias_image2d_;
        } else {
            throw std::runtime_error("Unsupported data types for OpenCLLinearOp Image Path.");
        }
        cl_mem a_img_mem = ocl_backend_->get_cl_mem(*A);
        cl_mem c_img_mem = ocl_backend_->get_cl_mem(*C);
        cl_sampler sampler = clCreateSampler(ocl_backend_->getContext(), CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
        check_cl_error(err, "clCreateSampler for LinearOp Image");
        int arg_idx = 0;
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_sampler), &sampler);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &a_img_mem);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &w_mem);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &bias_mem_arg);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &c_img_mem);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &M);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &K);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &N);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &H_size);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &K_b);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &has_bias_flag);
        const size_t global_work_size[2] = {(size_t)N / 4, (size_t)(B_size * H_size * M)};
        err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 2, nullptr, global_work_size, nullptr, 0, nullptr, &event);
        ocl_backend_->addProfilingEvent(this->name() + "_image", event);
        check_cl_error(err, "EnqueueNDRangeKernel Image GEMM");
        clReleaseSampler(sampler);
        tensorImage2Global(*C);
        tensorImage2Global(*A);
    } else {
        if (A->dtype() == MLLM_TYPE_F32 && W.dtype() == MLLM_TYPE_F32) {
            kernel_to_use = kernel_fp32_transb_bias_;
        } else if (A->dtype() == MLLM_TYPE_F16 && W.dtype() == MLLM_TYPE_F16) {
            kernel_to_use = kernel_fp16_transb_bias_;
        } else if (A->dtype() == MLLM_TYPE_F32 && W.dtype() == MLLM_TYPE_Q4_0) {
            kernel_to_use = kernel_fp32_q4_0_transb_bias_;
        } else if (A->dtype() == MLLM_TYPE_F16 && W.dtype() == MLLM_TYPE_Q4_0) {
            kernel_to_use = kernel_fp16_q4_0_transb_bias_;
        } else {
            throw std::runtime_error("Unsupported data types for OpenCLLinearOp.");
        }
        int arg_idx = 0;
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &a_mem);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &w_mem);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &bias_mem_arg);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &c_mem);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &M);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &K);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &N);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &H_size);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &K_b);
        clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &has_bias_flag);
        if (kernel_to_use == kernel_fp16_q4_0_transb_bias_ && ocl_backend_->has_fp16_support()) {
            const size_t TILE_M = 64;
            const size_t TILE_N = 64;
            const size_t THREADS_X = 8;
            const size_t THREADS_Y = 8;
            const size_t global_work_size[3] = {
                (size_t)ceil((float)N / TILE_N) * THREADS_X,
                (size_t)ceil((float)M / TILE_M) * THREADS_Y,
                (size_t)(B_size * H_size)};
            const size_t local_work_size[3] = {THREADS_X, THREADS_Y, 1};
            cl_event event;
            cl_int err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 3, nullptr, global_work_size, local_work_size, 0, nullptr, &event);
            ocl_backend_->addProfilingEvent(this->name() + "_tiled_q4", event);
            check_cl_error(err, "EnqueueNDRangeKernel tiled gemm_fp16_q4_0_transb_bias");

        } else {
            const size_t TILE_SIZE = 16;
            const size_t global_work_size[3] = {
                (size_t)(((N + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE),
                (size_t)(((M + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE),
                (size_t)(B_size * H_size)};
            const size_t local_work_size[3] = {TILE_SIZE, TILE_SIZE, 1};
            cl_event event;
            cl_int err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 3, nullptr, global_work_size, local_work_size, 0, nullptr, &event);
            ocl_backend_->addProfilingEvent(this->name(), event);
            check_cl_error(err, "EnqueueNDRangeKernel fused matmul_bias");
        }
    }

    return MLLM_NO_ERROR;
}

} // namespace mllm