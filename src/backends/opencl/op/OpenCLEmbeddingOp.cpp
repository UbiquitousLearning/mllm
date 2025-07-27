#include "OpenCLEmbeddingOp.hpp"
#include "Types.hpp"
// #include "utils/OpenCLTools.hpp"

namespace mllm {

OpenCLEmbeddingOp::OpenCLEmbeddingOp(Backend *bn, std::string name, int vocab_size, int hidden_size) :
    Op(bn, std::move(name)), vocab_size_(vocab_size), hidden_size_(hidden_size) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    const std::string kernel_path = "kernel/embedding.cl";
    cl_program program = ocl_backend_->getProgram(kernel_path);

    cl_int err;
    kernel_fp32_ = clCreateKernel(program, "embedding_fp32", &err);
    check_cl_error(err, "clCreateKernel for embedding_fp32");

    kernel_q4_0_ = clCreateKernel(program, "embedding_q4_0", &err);
    check_cl_error(err, "clCreateKernel for embedding_q4_0");

    kernel_q4_0_fp16_ = clCreateKernel(program, "embedding_q4_0_fp16", &err);
    check_cl_error(err, "clCreateKernel for embedding_q4_0_fp16");
}

OpenCLEmbeddingOp::~OpenCLEmbeddingOp() {
    if (kernel_fp32_) clReleaseKernel(kernel_fp32_);
    if (kernel_q4_0_) clReleaseKernel(kernel_q4_0_);
    if (kernel_q4_0_fp16_) clReleaseKernel(kernel_q4_0_fp16_);
}

ErrorCode OpenCLEmbeddingOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // 输出张量的形状是 [B, H, S, D]，其中 D 是 hidden_size
    // 注意：Embedding通常不关心H，这里假设H=1
    outputs[0]->reshape(inputs[0]->batch(), 1, inputs[0]->sequence(), hidden_size_);
    // Embedding的输出总是FP32
    outputs[0]->setDtype(inputs[0]->dtype());
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLEmbeddingOp::load(AbstructLoader &loader) {
    weight_.setBackend(Backend::global_backends[MLLM_CPU].get());
    // 从模型文件中加载权重
    weight_.setName(name() + ".weight");
    // 权重的形状是 [vocab_size, hidden_size]，我们用 BHSD 来模拟 [1, 1, vocab_size, hidden_size]
    weight_.reshape(1, 1, vocab_size_, hidden_size_);

    if (loader.getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        loader.load(&weight_);
    } else {
        // 如果模型文件中没有，可能需要一个默认的空权重
        weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
    }
    weight_.to(MLLM_OPENCL);
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLEmbeddingOp::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    return Op::free(inputs, outputs);
}

ErrorCode OpenCLEmbeddingOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // 确保所有张量都在OpenCL设备上
    inputs[0]->to(MLLM_OPENCL); // input_ids
    // 输出总是 FP32
    outputs[0]->to(MLLM_OPENCL);
    outputs[0]->setDtype(inputs[0]->dtype());
    outputs[0]->alloc();
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLEmbeddingOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];   // (B, 1, S, 1) or (B, S) -
    auto output = outputs[0]; // (B, 1, S, D)

    cl_kernel kernel_to_use = nullptr;
    if (weight_.dtype() == MLLM_TYPE_F32) {
        kernel_to_use = kernel_fp32_;
    } else if (weight_.dtype() == MLLM_TYPE_Q4_0) {
        if (output->dtype() == MLLM_TYPE_F16) {
            kernel_to_use = kernel_q4_0_fp16_; // 调用新的FP16输出内核
        } else {
            kernel_to_use = kernel_q4_0_; // 保留对FP32输出的兼容
        }
    } else {
        return NOT_SUPPORT;
    }

    cl_mem in_id_buf = ocl_backend_->get_cl_mem(*input);
    cl_mem weight_buf = ocl_backend_->get_cl_mem(weight_);
    cl_mem out_buf = ocl_backend_->get_cl_mem(*output);

    // input tensor (token_ids) is usually flat, e.g. (B*S)
    const int sequence_len = input->batch() * input->sequence();

    clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &in_id_buf);
    clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &weight_buf);
    clSetKernelArg(kernel_to_use, 2, sizeof(cl_mem), &out_buf);
    clSetKernelArg(kernel_to_use, 3, sizeof(int), &vocab_size_);
    clSetKernelArg(kernel_to_use, 4, sizeof(int), &hidden_size_);
    clSetKernelArg(kernel_to_use, 5, sizeof(int), &sequence_len);

    // 启动2D内核：
    // - 维度0 (X): 对应 hidden_size，每个工作项负责拷贝一个维度
    // - 维度1 (Y): 对应 token 数量 (B*S)，每个工作项负责处理一个 token
    const size_t global_work_size[2] = {(size_t)hidden_size_, (size_t)sequence_len};
    cl_event event;
    cl_int err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 2, nullptr, global_work_size, nullptr, 0, nullptr, &event);
    ocl_backend_->addProfilingEvent(this->name(), event);
    check_cl_error(err, "clEnqueueNDRangeKernel for Embedding");

    return MLLM_NO_ERROR;
}

} // namespace mllm