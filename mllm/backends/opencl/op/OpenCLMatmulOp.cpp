
#include "OpenCLMatmulOp.hpp"
#include "Types.hpp"
#include "utils/OpenCLTools.hpp"
#include <cassert>
#include <string>

namespace mllm {

// Constructor remains the same
OpenCLMatmulOp::OpenCLMatmulOp(Backend *bn, std::string name) :
    Op(bn, std::move(name)) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) {
        throw std::runtime_error("Backend for MatmulOp is not OpenCLBackend");
    }
    const std::string kernel_path = "kernel/matmul.cl";
    std::string build_options;
    if (ocl_backend_->has_fp16_support()) {
        build_options += "-DSUPPORTS_FP16";
    }

    cl_program program = ocl_backend_->getProgram(kernel_path, build_options);
    cl_int err;
    kernel_fp32_ = clCreateKernel(program, "gemm_fp32", &err);
    check_cl_error(err, "CreateKernel gemm_fp32");
    kernel_fp16_ = clCreateKernel(program, "gemm_fp16", &err);
    check_cl_error(err, "CreateKernel gemm_fp16");
    kernel_fp32_bhsd_ = clCreateKernel(program, "gemm_fp32_bhsd", &err);
    check_cl_error(err, "CreateKernel gemm_fp32_bhsd");
    kernel_fp16_bhsd_ = clCreateKernel(program, "gemm_fp16_bhsd", &err);
    check_cl_error(err, "CreateKernel gemm_fp16_bhsd");

    const std::string kernel_transb_path = "kernel/matmul_transb.cl";
    cl_program program_tansb = ocl_backend_->getProgram(kernel_transb_path, build_options);
    kernel_fp32_transb_ = clCreateKernel(program_tansb, "gemm_fp32_transb", &err);
    check_cl_error(err, "CreateKernel gemm_fp32_transb");
    kernel_fp16_transb_ = clCreateKernel(program_tansb, "gemm_fp16_transb", &err);
    check_cl_error(err, "CreateKernel gemm_fp16_transb");
    kernel_fp32_q4_0_transb = clCreateKernel(program_tansb, "gemm_fp32_q4_0_transb", &err);
    check_cl_error(err, "CreateKernel gemm_fp32_q4_0_transb");
    kernel_fp16_q4_0_transb = clCreateKernel(program_tansb, "gemm_fp16_q4_0_transb", &err);
    check_cl_error(err, "CreateKernel gemm_fp16_q4_0_transb");
}

// Destructor remains the same
OpenCLMatmulOp::~OpenCLMatmulOp() {
    if (kernel_fp32_) clReleaseKernel(kernel_fp32_);
    if (kernel_fp16_) clReleaseKernel(kernel_fp16_);
    if (kernel_fp32_bhsd_) clReleaseKernel(kernel_fp32_bhsd_);
    if (kernel_fp16_bhsd_) clReleaseKernel(kernel_fp16_bhsd_);
    if (kernel_fp32_transb_) clReleaseKernel(kernel_fp32_transb_);
    if (kernel_fp16_transb_) clReleaseKernel(kernel_fp16_transb_);
    if (kernel_fp32_q4_0_transb) clReleaseKernel(kernel_fp32_q4_0_transb);
    if (kernel_fp16_q4_0_transb) clReleaseKernel(kernel_fp16_q4_0_transb);
}

ErrorCode OpenCLMatmulOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto &A = inputs[0];
    auto &B = inputs[1];
    auto &C = outputs[0];

    int M = A->sequence();
    int K = A->dimension();
    int N;

    // 智能判断是标准乘法还是转置乘法，并设置标志
    // 标准: A(M,K) * B(K,N)  => A.dimension() == B.sequence()
    // 转置: A(M,K) * B_T(N,K) => A.dimension() == B.dimension()
    if (A->dimension() == B->sequence()) {
        // 标准乘法
        use_transb_ = false; // <--- 设置标志
        N = B->dimension();
    } else if (A->dimension() == B->dimension()) {
        // 转置乘法
        use_transb_ = true; // <--- 设置标志
        N = B->sequence();
    } else {
        // 不支持的矩阵乘法形状
        return NOT_SUPPORT;
    }
    assert(inputs[0]->ctype() == inputs[1]->ctype() && "Input tensors must have the same ctype");
    C->setCtype(inputs[0]->ctype());
    // 重塑输出张量 C
    C->reshape(A->batch(), A->head(), M, N);
    C->setDtype(A->dtype());
    return MLLM_NO_ERROR;
}

// setUp 函数保持不变，reshape 已确保输出张量维度正确，alloc 将分配正确的空间
ErrorCode OpenCLMatmulOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    inputs[0]->to(MLLM_OPENCL);
    inputs[1]->to(MLLM_OPENCL);
    outputs[0]->to(MLLM_OPENCL);
    outputs[0]->alloc();
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLMatmulOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto &A = inputs[0];
    auto &B = inputs[1];
    auto &C = outputs[0];

    cl_kernel kernel_to_use = nullptr;
    DataType in_type = A->dtype();
    DataType weight_type = B->dtype();

    // === 修改后的内核选择逻辑 ===
    if (use_transb_) {
        // 选择转置内核
        if (in_type == MLLM_TYPE_F32 && weight_type == MLLM_TYPE_F32)
            kernel_to_use = kernel_fp32_transb_;
        else if (in_type == MLLM_TYPE_F16 && weight_type == MLLM_TYPE_F16)
            kernel_to_use = kernel_fp16_transb_;
        else if (in_type == MLLM_TYPE_F32 && weight_type == MLLM_TYPE_Q4_0) {
            // assert(inputs[0]->head() == 1 && "fp32_q40 only support head==1");
            kernel_to_use = kernel_fp32_q4_0_transb;
        } else if (in_type == MLLM_TYPE_F16 && weight_type == MLLM_TYPE_Q4_0) {
            // assert(inputs[0]->head() == 1 && "fp16_q40 only support head==1");
            kernel_to_use = kernel_fp16_q4_0_transb;
        } else
            throw std::runtime_error("Unsupported data types for OpenCL Matmul operation.");
    } else {
        if (A->ctype() == BHSD) {
            if (in_type == MLLM_TYPE_F32 && weight_type == MLLM_TYPE_F32)
                kernel_to_use = kernel_fp32_bhsd_;
            else if (in_type == MLLM_TYPE_F16 && weight_type == MLLM_TYPE_F16)
                kernel_to_use = kernel_fp16_bhsd_;
            else
                throw std::runtime_error("Unsupported data types for OpenCL Matmul BHSD operation.");
        } else { // Default to BSHD
            if (in_type == MLLM_TYPE_F32 && weight_type == MLLM_TYPE_F32)
                kernel_to_use = kernel_fp32_;
            else if (in_type == MLLM_TYPE_F16 && weight_type == MLLM_TYPE_F16)
                kernel_to_use = kernel_fp16_;
            else
                throw std::runtime_error("Unsupported data types for OpenCL Matmul BSHD operation.");
        }
    }

    if (kernel_to_use == nullptr) {
        throw std::runtime_error("No suitable OpenCL Matmul kernel found for the given data types and shape.");
    }

    const int M = C->sequence();
    const int K = A->dimension();
    const int N = C->dimension();
    const int B_size = A->batch();
    const int H_size = A->head();
    const int K_b = (A->dimension() == B->sequence()) ? B->sequence() : B->dimension(); // K for B

    cl_mem a_mem = ocl_backend_->get_cl_mem(*A);
    cl_mem b_mem = ocl_backend_->get_cl_mem(*B);
    cl_mem c_mem = ocl_backend_->get_cl_mem(*C);

    int arg_idx = 0;
    clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &a_mem);
    clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &b_mem);
    clSetKernelArg(kernel_to_use, arg_idx++, sizeof(cl_mem), &c_mem);
    clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &M);
    clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &K);
    clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &N);
    clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &H_size);
    clSetKernelArg(kernel_to_use, arg_idx++, sizeof(int), &K_b);
    cl_event event;

    // if (ocl_backend_->has_fp16_support() && !use_transb_ && A->ctype() == BHSD && in_type == MLLM_TYPE_F16 && weight_type == MLLM_TYPE_F16) {
    //     const int TILE_M = 8;
    //     const int TILE_N = 4;
    //     const size_t global_work_size[3] = {
    //         (size_t)((N + TILE_N - 1) / TILE_N),
    //         (size_t)((M + TILE_M - 1) / TILE_M),
    //         (size_t)(B_size * H_size)};
    //     const size_t *local_work_size = nullptr;
    //     cl_int err = clEnqueueNDRangeKernel(
    //         ocl_backend_->getQueue(), kernel_to_use, 3, nullptr,
    //         global_work_size,
    //         local_work_size,
    //         0, nullptr, &event);
    //     check_cl_error(err, "EnqueueNDRangeKernel Tiled Matmul 3D");
    // } else {
    const int TILE_SIZE = 16;
    const size_t global_work_size[3] = {
        (size_t)(((N + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE),
        (size_t)(((M + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE),
        (size_t)(B_size * H_size)};
    const size_t local_work_size[3] = {TILE_SIZE, TILE_SIZE, 1};

    cl_int err = clEnqueueNDRangeKernel(
        ocl_backend_->getQueue(), kernel_to_use, 3, nullptr,
        global_work_size,
        local_work_size,
        0, nullptr, &event);

    check_cl_error(err, "EnqueueNDRangeKernel Tiled Matmul 3D");
    // }

    ocl_backend_->addProfilingEvent(this->name() + "mat_mul", event);

    return MLLM_NO_ERROR;
}
} // namespace mllm
