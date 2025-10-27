#include "OpenCLSoftMaxOp.hpp"
#include "Types.hpp"
#include "utils/OpenCLTools.hpp"

namespace mllm {

OpenCLSoftMaxOp::OpenCLSoftMaxOp(Backend *bn, std::string name, int axis, bool do_causal_mask) :
    Op(bn, std::move(name)), axis_(axis), do_causal_mask_(do_causal_mask) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    if (axis_ == DIMENSION) {
        const std::string kernel_path = "kernel/softmax.cl";

        std::string build_options;
        if (ocl_backend_->has_fp16_support()) {
            build_options += " -DSUPPORTS_FP16";
        }

        cl_program program = ocl_backend_->getProgram(kernel_path, build_options);

        cl_int err;

        kernel_fp32_d_ = clCreateKernel(program, "softmax_f32_along_d", &err);
        check_cl_error(err, "clCreateKernel for softmax_f32_along_d");

        // 仅当硬件支持FP16时，才创建FP16内核
        // if (ocl_backend_->has_fp16_support()) {
        kernel_fp16_d_ = clCreateKernel(program, "softmax_fp16_along_d", &err);
        check_cl_error(err, "clCreateKernel for softmax_fp16_along_d");
        // }
    }
}

OpenCLSoftMaxOp::~OpenCLSoftMaxOp() {
    if (kernel_fp32_d_) clReleaseKernel(kernel_fp32_d_);
    if (kernel_fp16_d_) clReleaseKernel(kernel_fp16_d_);
}

ErrorCode OpenCLSoftMaxOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->setCtype(inputs[0]->ctype());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLSoftMaxOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // 确保输入在设备上，并为输出分配内存
    for (auto &input : inputs) {
        input->to(MLLM_OPENCL);
    }
    outputs[0]->setDtype(inputs[0]->dtype());
    outputs[0]->alloc();
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLSoftMaxOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (axis_ != DIMENSION) {
        throw std::runtime_error("OpenCLSoftMaxOp currently only supports axis=DIMENSION");
    }

    auto input = inputs[0];
    auto output = outputs[0];

    cl_kernel kernel_to_use = nullptr;
    if (input->dtype() == MLLM_TYPE_F32) {
        kernel_to_use = kernel_fp32_d_;
    } else if (input->dtype() == MLLM_TYPE_F16) {
        // 如果fp16内核未创建成功（因为硬件不支持），这里会是nullptr
        if (kernel_fp16_d_ == nullptr) {
            throw std::runtime_error("FP16 Softmax kernel is not available on this device.");
        }
        kernel_to_use = kernel_fp16_d_;
    } else {
        return NOT_SUPPORT;
    }

    const int B = input->batch();
    const int H = input->head();
    const int S = input->sequence();
    const int D = input->dimension();
    int do_causal_mask_int = do_causal_mask_ ? 1 : 0;
    if (input->sequence() == 1) {
        do_causal_mask_int = 0;
    }

    cl_mem src_buf = ocl_backend_->get_cl_mem(*input);
    cl_mem dst_buf = ocl_backend_->get_cl_mem(*output);

    // 设置所有内核参数
    clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &src_buf);
    clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &dst_buf);
    clSetKernelArg(kernel_to_use, 2, sizeof(int), &B);
    clSetKernelArg(kernel_to_use, 3, sizeof(int), &H);
    clSetKernelArg(kernel_to_use, 4, sizeof(int), &S);
    clSetKernelArg(kernel_to_use, 5, sizeof(int), &D);
    clSetKernelArg(kernel_to_use, 6, sizeof(int), &do_causal_mask_int);

    // --- 核心修正：修改内核启动配置 ---
    const size_t total_rows = (size_t)B * H * S;
    const size_t local_work_size = 256; // 每个工作组的线程数，必须与 .cl 文件中的 SOFTMAX_BLOCK_SIZE 一致

    // 全局工作大小 = 总行数 * 每个工作组的线程数
    // 这会启动 total_rows 个工作组，每个组有 local_work_size 个线程
    const size_t global_work_size = total_rows * local_work_size;
    cl_event event;
    cl_int err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 1, nullptr, &global_work_size, &local_work_size, 0, nullptr, &event);
    ocl_backend_->addProfilingEvent(this->name(), event);
    check_cl_error(err, "clEnqueueNDRangeKernel for SoftMax");

    return MLLM_NO_ERROR;
}

} // namespace mllm