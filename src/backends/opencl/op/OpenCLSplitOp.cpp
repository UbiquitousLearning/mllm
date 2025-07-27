#include "OpenCLSplitOp.hpp"
#include "Types.hpp"
// #include "utils/OpenCLTools.hpp"
// #include <numeric>

namespace mllm {

OpenCLSplitOp::OpenCLSplitOp(Backend *bn, std::string name, int num_splits, const std::vector<int> &each_dims, Chl split_dim, int head_size) :
    Op(bn, std::move(name)), num_splits_(num_splits), each_dims_(each_dims), split_dim_(split_dim), head_size_(head_size) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    const std::string kernel_path = "kernel/split.cl";
    cl_program program = ocl_backend_->getProgram(kernel_path);

    cl_int err;
    kernel_fp32_ = clCreateKernel(program, "split_fp32", &err);
    check_cl_error(err, "clCreateKernel for split_fp32");

    kernel_fp16_ = clCreateKernel(program, "split_fp16", &err);
    check_cl_error(err, "clCreateKernel for split_fp16");
}

OpenCLSplitOp::~OpenCLSplitOp() {
    if (kernel_fp32_) clReleaseKernel(kernel_fp32_);
    if (kernel_fp16_) clReleaseKernel(kernel_fp16_);
}

ErrorCode OpenCLSplitOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    int split_dim_size = 0;
    for (const auto &dim : each_dims_) {
        split_dim_size += dim;
    }

    for (int i = 0; i < num_splits_; ++i) {
        switch (split_dim_) {
        case Chl::HEAD:
            outputs[i]->reshape(input->batch(), each_dims_[i], input->sequence(), input->dimension());
            break;
        case Chl::SEQUENCE:
            outputs[i]->reshape(input->batch(), input->head(), each_dims_[i], input->dimension());
            break;
        case Chl::DIMENSION:
            outputs[i]->reshape(input->batch(), input->head(), input->sequence(), each_dims_[i]);
            break;
        default:
            return NOT_SUPPORT;
        }
        outputs[i]->setDtype(input->dtype());
    }
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLSplitOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    inputs[0]->to(MLLM_OPENCL);
    for (auto &output : outputs) {
        output->setDtype(inputs[0]->dtype());
        output->to(MLLM_OPENCL);
        output->alloc();
    }
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLSplitOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];

    cl_kernel kernel_to_use = (input->dtype() == MLLM_TYPE_F32) ? kernel_fp32_ : kernel_fp16_;

    cl_mem in_buf = ocl_backend_->get_cl_mem(*input);

    std::vector<int> offsets(num_splits_, 0);
    for (int i = 1; i < num_splits_; ++i) {
        offsets[i] = offsets[i - 1] + each_dims_[i - 1];
    }

    int outer_size = 1;
    int inner_size = 1;

    switch (split_dim_) {
    case Chl::HEAD:
        outer_size = input->batch();
        inner_size = input->sequence() * input->dimension();
        break;
    case Chl::SEQUENCE:
        outer_size = input->batch() * input->head();
        inner_size = input->dimension();
        break;
    case Chl::DIMENSION:
        outer_size = input->batch() * input->head() * input->sequence();
        inner_size = 1;
        break;
    default:
        return NOT_SUPPORT;
    }
    int dims = input->shape(split_dim_);

    for (int i = 0; i < num_splits_; ++i) {
        cl_mem out_buf = ocl_backend_->get_cl_mem(*outputs[i]);
        int split_dim_size = each_dims_[i];
        int offset = offsets[i];
        if (inner_size == 0 || split_dim_size == 0 || outer_size == 0) {
            continue;
        }
        clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &in_buf);
        clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &out_buf);
        clSetKernelArg(kernel_to_use, 2, sizeof(int), &outer_size);
        clSetKernelArg(kernel_to_use, 3, sizeof(int), &split_dim_size);
        clSetKernelArg(kernel_to_use, 4, sizeof(int), &inner_size);
        clSetKernelArg(kernel_to_use, 5, sizeof(int), &offset);
        clSetKernelArg(kernel_to_use, 6, sizeof(int), &dims);

        const size_t global_work_size[3] = {(size_t)inner_size, (size_t)split_dim_size, (size_t)outer_size};
        cl_event event;
        cl_int err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 3, nullptr, global_work_size, nullptr, 0, nullptr, &event);
        ocl_backend_->addProfilingEvent(this->name() + "split", event);
        if (err != CL_SUCCESS) {
            std::cout << "clEnqueueNDRangeKernel error: split" << inner_size << " " << split_dim_size << " " << outer_size << std::endl;
        }
        check_cl_error(err, "clEnqueueNDRangeKernel for Split");
    }

    return MLLM_NO_ERROR;
}

} // namespace mllm