#include "OpenCLTopkOp.hpp"
#include "Types.hpp"
// #include "utils/OpenCLTools.hpp"

namespace mllm {

OpenCLTopkOp::OpenCLTopkOp(Backend *bn, std::string name, int k, Chl dim) :
    Op(bn, std::move(name)), k_(k), dim_(dim) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    const std::string kernel_path = "kernel/topk.cl";
    cl_program program = ocl_backend_->getProgram(kernel_path);

    cl_int err;
    kernel_fp32_ = clCreateKernel(program, "topk_fp32", &err);
    check_cl_error(err, "clCreateKernel for topk_fp32");

    kernel_fp16_ = clCreateKernel(program, "topk_fp16", &err);
    check_cl_error(err, "clCreateKernel for topk_fp16");
}

OpenCLTopkOp::~OpenCLTopkOp() {
    if (kernel_fp32_) clReleaseKernel(kernel_fp32_);
    if (kernel_fp16_) clReleaseKernel(kernel_fp16_);
}

ErrorCode OpenCLTopkOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (dim_ != DIMENSION) {
        return NOT_SUPPORT;
    }
    assert(outputs.size() == 2); // topk returns values and indices

    auto input = inputs[0];
    auto values_out = outputs[0];
    auto indices_out = outputs[1];

    values_out->reshape(input->batch(), input->head(), input->sequence(), k_);
    indices_out->reshape(input->batch(), input->head(), input->sequence(), k_);

    values_out->setDtype(input->dtype());
    // 虽然索引通常是整数，但为了简化和保持一致性，这里也使用与输入相同的数据类型
    indices_out->setDtype(input->dtype());

    return MLLM_NO_ERROR;
}

ErrorCode OpenCLTopkOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    inputs[0]->to(MLLM_OPENCL);
    for (auto &output : outputs) {
        output->setDtype(inputs[0]->dtype());
        output->alloc();
    }
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLTopkOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto values_out = outputs[0];
    auto indices_out = outputs[1];

    cl_kernel kernel_to_use = (input->dtype() == MLLM_TYPE_F32) ? kernel_fp32_ : kernel_fp16_;

    cl_mem in_buf = ocl_backend_->get_cl_mem(*input);
    cl_mem values_buf = ocl_backend_->get_cl_mem(*values_out);
    cl_mem indices_buf = ocl_backend_->get_cl_mem(*indices_out);

    const int B = input->batch();
    const int H = input->head();
    const int S = input->sequence();
    const int D = input->dimension();

    clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &in_buf);
    clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &values_buf);
    clSetKernelArg(kernel_to_use, 2, sizeof(cl_mem), &indices_buf);
    clSetKernelArg(kernel_to_use, 3, sizeof(int), &D);
    clSetKernelArg(kernel_to_use, 4, sizeof(int), &k_);

    // 启动一个工作组来处理每一行
    const size_t total_rows = (size_t)B * H * S;
    const size_t local_work_size = 256; // 必须是2的幂，与内核中的 WG_SIZE 保持一致
    const size_t global_work_size = total_rows * local_work_size;
    cl_event event;
    cl_int err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 1, nullptr,
                                        &global_work_size, &local_work_size, 0, nullptr, &event);
    ocl_backend_->addProfilingEvent(this->name() + "topk", event);
    check_cl_error(err, "clEnqueueNDRangeKernel for TopK");

    return MLLM_NO_ERROR;
}

} // namespace mllm