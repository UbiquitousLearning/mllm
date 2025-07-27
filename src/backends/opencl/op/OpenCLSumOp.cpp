#include "OpenCLSumOp.hpp"
#include "Types.hpp"
// #include "utils/OpenCLTools.hpp"

namespace mllm {

OpenCLSumOp::OpenCLSumOp(Backend *bn, std::string name, Chl axis) :
    Op(bn, std::move(name)), axis_(axis) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    const std::string kernel_path = "kernel/sum.cl";

    std::string build_options;
    if (ocl_backend_->has_fp16_support()) {
        build_options += " -DSUPPORTS_FP16";
    }

    cl_program program = ocl_backend_->getProgram(kernel_path, build_options);

    cl_int err;
    kernel_fp32_ = clCreateKernel(program, "sum_fp32", &err);
    check_cl_error(err, "clCreateKernel for sum_fp32");

    kernel_fp16_ = clCreateKernel(program, "sum_fp16", &err);
    check_cl_error(err, "clCreateKernel for sum_fp16");
}

OpenCLSumOp::~OpenCLSumOp() {
    if (kernel_fp32_) clReleaseKernel(kernel_fp32_);
    if (kernel_fp16_) clReleaseKernel(kernel_fp16_);
}

ErrorCode OpenCLSumOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    int batch = inputs[0]->batch();
    int head = inputs[0]->head();
    int sequence = inputs[0]->sequence();
    int dimension = inputs[0]->dimension();

    switch (axis_) {
    case BATCH: batch = 1; break;
    case HEAD: head = 1; break;
    case SEQUENCE: sequence = 1; break;
    case DIMENSION: dimension = 1; break;
    default: break;
    }

    outputs[0]->reshape(batch, head, sequence, dimension);
    outputs[0]->setDtype(inputs[0]->dtype());
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLSumOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    inputs[0]->to(MLLM_OPENCL);
    outputs[0]->setDtype(inputs[0]->dtype());
    outputs[0]->alloc();
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLSumOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto output = outputs[0];

    cl_kernel kernel_to_use = (input->dtype() == MLLM_TYPE_F32) ? kernel_fp32_ : kernel_fp16_;

    int outer_size = 1;
    int inner_size = 1;
    int reduce_size = 1;

    // 1. Get the actual memory index for the reduction axis
    const int axis_mem_idx = input->chls().at(axis_);

    // 2. Get the total number of axes from the shape vector's size
    const int num_axes = input->shape().size();

    // 3. Calculate outer_size (product of dimensions before the reduction axis)
    for (int i = 0; i < axis_mem_idx; ++i) {
        outer_size *= input->legacyShape(i);
    }

    // 4. Get the size of the reduction dimension
    reduce_size = input->legacyShape(axis_mem_idx);

    // 5. Calculate inner_size (product of dimensions after the reduction axis)
    for (int i = axis_mem_idx + 1; i < num_axes; ++i) {
        inner_size *= input->legacyShape(i);
    }

    cl_mem in_buf = ocl_backend_->get_cl_mem(*input);
    cl_mem out_buf = ocl_backend_->get_cl_mem(*output);

    clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &in_buf);
    clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &out_buf);
    clSetKernelArg(kernel_to_use, 2, sizeof(int), &outer_size);
    clSetKernelArg(kernel_to_use, 3, sizeof(int), &inner_size);
    clSetKernelArg(kernel_to_use, 4, sizeof(int), &reduce_size);

    // 1. 将 local_work_size 定义为二维数组
    const size_t local_work_size[2] = {256, 1}; // 第一个维度是规约并行度，第二个是1

    // 2. 确保 global_work_size 的每个维度都是 local_work_size 对应维度的整数倍
    const size_t global_work_size[2] = {
        (size_t)inner_size * local_work_size[0],
        (size_t)outer_size * local_work_size[1] // outer_size * 1
    };
    // ==============================================================

    cl_event event;
    // 3. 将二维的 local_work_size 数组传递给内核
    cl_int err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 2, nullptr,
                                        global_work_size, local_work_size, 0, nullptr, &event);
    ocl_backend_->addProfilingEvent(this->name() + "sum", event);
    check_cl_error(err, "clEnqueueNDRangeKernel for Sum");

    return MLLM_NO_ERROR;
}

} // namespace mllm