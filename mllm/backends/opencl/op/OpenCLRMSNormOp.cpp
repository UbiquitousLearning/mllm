// OpenCLRMSNormOp.cpp

#include "OpenCLRMSNormOp.hpp"
#include "Types.hpp"
// #include "utils/OpenCLTools.hpp"

namespace mllm {

// 构造函数
OpenCLRMSNormOp::OpenCLRMSNormOp(Backend *bn, std::string name, int normSize, float epsilon, bool add_unit_offset) :
    Op(bn, std::move(name)), normSize_(normSize), epsilon_(epsilon), add_unit_offset_(add_unit_offset) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    const std::string kernel_path = "kernel/rmsnorm.cl";
    std::string build_options;
    if (ocl_backend_->has_fp16_support()) {
        build_options += " -DSUPPORTS_FP16";
    }

    cl_program program = ocl_backend_->getProgram(kernel_path, build_options);

    cl_int err;
    // ✨ 修改: 创建两个内核
    kernel_fp32_ = clCreateKernel(program, "rmsnorm_f32_q4", &err);
    check_cl_error(err, "clCreateKernel for rmsnorm_f32_q4");

    kernel_fp16_ = clCreateKernel(program, "rmsnorm_f16_q4", &err);
    check_cl_error(err, "clCreateKernel for rmsnorm_f16_q4");
}

// ✨ 新增: 自定义析构函数
OpenCLRMSNormOp::~OpenCLRMSNormOp() {
    if (kernel_fp32_) clReleaseKernel(kernel_fp32_);
    if (kernel_fp16_) clReleaseKernel(kernel_fp16_);
}

// reshape, load, free, setUp 函数保持不变
ErrorCode OpenCLRMSNormOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(normSize_ == inputs[0]->dimension());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    // ✨ 新增: 确保输出类型与输入一致
    outputs[0]->setDtype(inputs[0]->dtype());
    return Op::reshape(inputs, outputs);
}
ErrorCode OpenCLRMSNormOp::load(AbstructLoader &loader) {
    weight_.setBackend(Backend::global_backends[MLLM_CPU].get());
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, 1, normSize_);
    if (loader.getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        loader.load(&weight_);
        weight_.to(MLLM_OPENCL);
    } else {
        weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
        weight_.to(MLLM_OPENCL);
    }
    return Op::load(loader);
}
ErrorCode OpenCLRMSNormOp::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    return Op::free(inputs, outputs);
}
ErrorCode OpenCLRMSNormOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    inputs[0]->to(MLLM_OPENCL);
    outputs[0]->setDtype(inputs[0]->dtype());
    outputs[0]->alloc();
    outputs[0]->to(MLLM_OPENCL);
    return MLLM_NO_ERROR;
}

// execute 函数
ErrorCode OpenCLRMSNormOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto output = outputs[0];

    // ✨ 修改: 根据输入类型选择内核
    cl_kernel kernel_to_use = nullptr;
    if (input->dtype() == MLLM_TYPE_F32) {
        kernel_to_use = kernel_fp32_;
    } else if (input->dtype() == MLLM_TYPE_F16) {
        kernel_to_use = kernel_fp16_;
    } else {
        return NOT_SUPPORT;
    }

    const int D = input->dimension();
    const int weight_is_q4 = (weight_.dtype() == MLLM_TYPE_Q4_0) ? 1 : 0;
    const int add_unit_offset_int = add_unit_offset_ ? 1 : 0;

    cl_mem src_buf = ocl_backend_->get_cl_mem(*input);
    cl_mem dst_buf = ocl_backend_->get_cl_mem(*output);
    cl_mem w_buf = ocl_backend_->get_cl_mem(weight_);

    clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &src_buf);
    clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &dst_buf);
    clSetKernelArg(kernel_to_use, 2, sizeof(cl_mem), &w_buf);
    clSetKernelArg(kernel_to_use, 3, sizeof(int), &weight_is_q4);
    clSetKernelArg(kernel_to_use, 4, sizeof(int), &D);
    clSetKernelArg(kernel_to_use, 5, sizeof(float), &epsilon_);
    clSetKernelArg(kernel_to_use, 6, sizeof(int), &add_unit_offset_int);

    const size_t total_rows = (size_t)input->batch() * input->head() * input->sequence();
    const size_t local_work_size = 256;
    const size_t global_work_size = total_rows * local_work_size;

    cl_event event;
    cl_int err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, &event);
    ocl_backend_->addProfilingEvent(this->name(), event);
    check_cl_error(err, "clEnqueueNDRangeKernel for RMSNorm");

    return MLLM_NO_ERROR;
}

} // namespace mllm