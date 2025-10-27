#include "OpenCLClipTensorOp.hpp"
#include "Types.hpp"
// #include "utils/OpenCLTools.hpp"

namespace mllm {

OpenCLClipTensorOp::OpenCLClipTensorOp(Backend *bn, std::string name, Chl dim) :
    Op(bn, std::move(name)), dim_(dim) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    const std::string kernel_path = "kernel/cliptensor.cl";
    cl_program program = ocl_backend_->getProgram(kernel_path);

    cl_int err;
    kernel_seq_fp32_ = clCreateKernel(program, "clip_sequence_fp32", &err);
    check_cl_error(err, "clCreateKernel for clip_sequence_fp32");
    kernel_seq_fp16_ = clCreateKernel(program, "clip_sequence_fp16", &err);
    check_cl_error(err, "clCreateKernel for clip_sequence_fp16");

    kernel_dim_fp32_ = clCreateKernel(program, "clip_dimension_fp32", &err);
    check_cl_error(err, "clCreateKernel for clip_dimension_fp32");
    kernel_dim_fp16_ = clCreateKernel(program, "clip_dimension_fp16", &err);
    check_cl_error(err, "clCreateKernel for clip_dimension_fp16");
}

OpenCLClipTensorOp::~OpenCLClipTensorOp() {
    if (kernel_seq_fp32_) clReleaseKernel(kernel_seq_fp32_);
    if (kernel_seq_fp16_) clReleaseKernel(kernel_seq_fp16_);
    if (kernel_dim_fp32_) clReleaseKernel(kernel_dim_fp32_);
    if (kernel_dim_fp16_) clReleaseKernel(kernel_dim_fp16_);
}

ErrorCode OpenCLClipTensorOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto indices = inputs[1];
    auto output = outputs[0];

    if (dim_ == SEQUENCE) {
        int new_seq = indices->dimension(); // Indices are 1D, stored in the dimension field
        output->reshape(input->batch(), input->head(), new_seq, input->dimension());
    } else if (dim_ == DIMENSION) {
        int new_dim = indices->dimension();
        output->reshape(input->batch(), input->head(), input->sequence(), new_dim);
    } else {
        return NOT_SUPPORT;
    }
    output->setDtype(input->dtype());
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLClipTensorOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    inputs[0]->to(MLLM_OPENCL);
    inputs[1]->to(MLLM_OPENCL); // Indices tensor also needs to be on the device
    outputs[0]->setDtype(inputs[0]->dtype());
    outputs[0]->alloc();
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLClipTensorOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto indices = inputs[1];
    auto output = outputs[0];

    if (input->ctype() != BSHD || output->ctype() != BSHD) {
        return NOT_SUPPORT;
    }

    cl_kernel kernel_to_use = nullptr;
    cl_event event;
    cl_int err;

    cl_mem in_buf = ocl_backend_->get_cl_mem(*input);
    cl_mem indices_buf = ocl_backend_->get_cl_mem(*indices);
    cl_mem out_buf = ocl_backend_->get_cl_mem(*output);

    if (dim_ == SEQUENCE) {
        kernel_to_use = (input->dtype() == MLLM_TYPE_F32) ? kernel_seq_fp32_ : kernel_seq_fp16_;

        const int B = input->batch();
        const int H = input->head();
        const int S_in = input->sequence();
        const int D = input->dimension();
        const int S_out = output->sequence();

        clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &in_buf);
        clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &indices_buf);
        clSetKernelArg(kernel_to_use, 2, sizeof(cl_mem), &out_buf);
        clSetKernelArg(kernel_to_use, 3, sizeof(int), &B);
        clSetKernelArg(kernel_to_use, 4, sizeof(int), &H);
        clSetKernelArg(kernel_to_use, 5, sizeof(int), &S_in);
        clSetKernelArg(kernel_to_use, 6, sizeof(int), &D);
        clSetKernelArg(kernel_to_use, 7, sizeof(int), &S_out);

        const size_t global_work_size[3] = {(size_t)D, (size_t)H, (size_t)B * S_out};
        err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 3, nullptr, global_work_size, nullptr, 0, nullptr, &event);
        check_cl_error(err, "clEnqueueNDRangeKernel for Clip Sequence");

    } else if (dim_ == DIMENSION) {
        kernel_to_use = (input->dtype() == MLLM_TYPE_F32) ? kernel_dim_fp32_ : kernel_dim_fp16_;

        const int B = input->batch();
        const int H = input->head();
        const int S = input->sequence();
        const int D_in = input->dimension();
        const int D_out = output->dimension();

        clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &in_buf);
        clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &indices_buf);
        clSetKernelArg(kernel_to_use, 2, sizeof(cl_mem), &out_buf);
        clSetKernelArg(kernel_to_use, 3, sizeof(int), &B);
        clSetKernelArg(kernel_to_use, 4, sizeof(int), &H);
        clSetKernelArg(kernel_to_use, 5, sizeof(int), &S);
        clSetKernelArg(kernel_to_use, 6, sizeof(int), &D_in);
        clSetKernelArg(kernel_to_use, 7, sizeof(int), &D_out);

        const size_t global_work_size[3] = {(size_t)D_out, (size_t)S, (size_t)B * H};
        err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 3, nullptr, global_work_size, nullptr, 0, nullptr, &event);
        check_cl_error(err, "clEnqueueNDRangeKernel for Clip Dimension");

    } else {
        return NOT_SUPPORT;
    }

    return MLLM_NO_ERROR;
}

} // namespace mllm