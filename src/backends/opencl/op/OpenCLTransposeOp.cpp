#include "OpenCLTransposeOp.hpp"
#include "Types.hpp"
#include "utils/OpenCLTools.hpp"
#include <iostream>

namespace mllm {

OpenCLTransposeOp::OpenCLTransposeOp(Backend *bn, std::string name, const vector<std::pair<Chl, Chl>> &axiss) :
    Op(bn, std::move(name)), axiss_(axiss) {
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    const std::string kernel_path = "kernel/transpose.cl";
    cl_program program = ocl_backend_->getProgram(kernel_path);

    cl_int err;
    kernel_fp32_2d_ = clCreateKernel(program, "transpose_float_2d", &err);
    check_cl_error(err, "clCreateKernel for transpose_float_2d");

    kernel_fp16_2d_ = clCreateKernel(program, "transpose_fp16_2d", &err);
    check_cl_error(err, "clCreateKernel for transpose_fp16_2d");

    kernel_fp32_bshd_ = clCreateKernel(program, "transpose_bshd2bhsd_fp32", &err);
    check_cl_error(err, "clCreateKernel for transpose_bshd2bhsd_fp32");
    kernel_fp16_bshd_ = clCreateKernel(program, "transpose_bshd2bhsd_fp16", &err);
    check_cl_error(err, "clCreateKernel for transpose_bshd2bhsd_fp16");

    kernel_fp32_bhsd_ = clCreateKernel(program, "transpose_bhsd2bshd_fp32", &err);
    check_cl_error(err, "clCreateKernel for transpose_bhsd2bshd_fp32");
    kernel_fp16_bhsd_ = clCreateKernel(program, "transpose_bhsd2bshd_fp16", &err);
    check_cl_error(err, "clCreateKernel for transpose_bhsd2bshd_fp16");
}

OpenCLTransposeOp::~OpenCLTransposeOp() {
    if (kernel_fp32_2d_) clReleaseKernel(kernel_fp32_2d_);
    if (kernel_fp16_2d_) clReleaseKernel(kernel_fp16_2d_);
    if (kernel_fp32_bshd_) clReleaseKernel(kernel_fp32_bshd_);
    if (kernel_fp16_bshd_) clReleaseKernel(kernel_fp16_bshd_);
}

ErrorCode OpenCLTransposeOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // Transpose on BHSD -> BHDS (swapping Sequence and Dimension)
    if (axiss_.size() == 1 && axiss_[0].first == SEQUENCE && axiss_[0].second == DIMENSION && inputs[0]->ctype() == BHSD) {
        outputs[0]->setCtype(BHSD);
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->dimension(), inputs[0]->sequence());
    } else if (axiss_.size() == 1 && axiss_[0].first == HEAD && axiss_[0].second == SEQUENCE) {
        // H,S 转置 (BSHD -> BHSD)
        auto input = inputs[0];
        auto output = outputs[0];
        // 这一步是元数据变换，定义逻辑形状
        output->transCopyShape(input->shape());
        output->chls() = input->chls();
        std::swap(output->chls()[HEAD], output->chls()[SEQUENCE]);
        output->changeCtype(input->shape().size());
        // 物理形状与输入保持一致，因为将在execute中进行物理拷贝
        output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
    } else {
        std::cerr << "error reshape" << std::endl;
    }
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLTransposeOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // 确保输入和输出的数据类型一致
    outputs[0]->setDtype(inputs[0]->dtype());

    // 将输入张量的数据转移到OpenCL设备
    for (auto &input : inputs) {
        input->to(MLLM_OPENCL);
    }

    // 为输出张量分配内存，您的框架会根据分页策略进行管理
    outputs[0]->alloc();
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLTransposeOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto output = outputs[0];

    cl_mem parent_src_buffer = ocl_backend_->get_cl_mem(*input);
    cl_mem parent_dst_buffer = ocl_backend_->get_cl_mem(*output);
    cl_int err;

    // === 新增: 处理 H,S 轴转置的逻辑分支 ===
    if (axiss_.size() == 1 && axiss_[0].first == HEAD && axiss_[0].second == SEQUENCE) {
        cl_kernel kernel_to_use = nullptr;
        if (input->ctype() == BSHD) { // BSHD -> BHSD
            if (input->dtype() == MLLM_TYPE_F32) {
                kernel_to_use = kernel_fp32_bshd_;
            } else {
                kernel_to_use = kernel_fp16_bshd_;
            }
        } else if (input->ctype() == BHSD) { // BHSD -> BSHD
            if (input->dtype() == MLLM_TYPE_F32) {
                kernel_to_use = kernel_fp32_bhsd_;
            } else {
                kernel_to_use = kernel_fp16_bhsd_;
            }
        } else {
            return NOT_SUPPORT;
        }
        const int B = input->batch();
        const int H = input->head();
        const int S = input->sequence();
        const int D = input->dimension();

        cl_mem src_buf = ocl_backend_->get_cl_mem(*input);
        cl_mem dst_buf = ocl_backend_->get_cl_mem(*output);

        clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &src_buf);
        clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &dst_buf);
        clSetKernelArg(kernel_to_use, 2, sizeof(int), &B);
        clSetKernelArg(kernel_to_use, 3, sizeof(int), &H);
        clSetKernelArg(kernel_to_use, 4, sizeof(int), &S);
        clSetKernelArg(kernel_to_use, 5, sizeof(int), &D);

        const size_t global_work_size[3] = {(size_t)D, (size_t)S, (size_t)H * B};
        cl_event event;
        clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 3, nullptr, global_work_size, nullptr, 0, nullptr, &event);
        ocl_backend_->addProfilingEvent(this->name() + "transpose", event);
    }
    // === 处理 S,D 轴转置的逻辑保持不变 ===
    else if (axiss_.size() == 1 && axiss_[0].first == SEQUENCE && axiss_[0].second == DIMENSION && inputs[0]->ctype() == BHSD) {
        cl_kernel kernel_to_use = nullptr;
        size_t element_size = 0;
        if (input->dtype() == MLLM_TYPE_F32) {
            kernel_to_use = kernel_fp32_2d_;
            element_size = sizeof(float);
        } else if (input->dtype() == MLLM_TYPE_F16) {
            kernel_to_use = kernel_fp16_2d_;
            element_size = sizeof(mllm_fp16_t);
        } else {
            return NOT_SUPPORT;
        }
        for (int b = 0; b < input->batch(); ++b) {
            for (int h = 0; h < input->head(); ++h) {
                const int S = input->sequence();
                const int D = input->dimension();

                // 计算偏移量（以元素数量为单位）
                int src_offset_elements = (b * input->head() + h) * S * D;
                int dst_offset_elements = (b * output->head() + h) * D * S;

                // 不再创建 SubBuffer
                // clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &src_sub_buffer);
                // clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &dst_sub_buffer);

                // 传递父缓冲区和偏移量
                clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &parent_src_buffer);
                clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &parent_dst_buffer);
                clSetKernelArg(kernel_to_use, 2, sizeof(int), &S);
                clSetKernelArg(kernel_to_use, 3, sizeof(int), &D);
                clSetKernelArg(kernel_to_use, 4, sizeof(int), &src_offset_elements);
                clSetKernelArg(kernel_to_use, 5, sizeof(int), &dst_offset_elements);

                const size_t block_dim = 16;
                const size_t global_work_size[2] = {
                    (size_t)D + ((block_dim - (size_t)D % block_dim) % block_dim),
                    (size_t)S + ((block_dim - (size_t)S % block_dim) % block_dim)};
                const size_t local_work_size[2] = {block_dim, block_dim};
                cl_event event;
                clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 2, nullptr, global_work_size, local_work_size, 0, nullptr, &event);
                ocl_backend_->addProfilingEvent(this->name() + "transpose", event);
            }
        }
    } else {
        std::cerr << "OpenCLTransposeOp execute error: unsupported transpose axis or ctype" << std::endl;
        return NOT_SUPPORT;
    }

    return MLLM_NO_ERROR;
}

} // namespace mllm