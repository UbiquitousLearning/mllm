// 文件: OpenCLKVCacheOp.cpp

#include "OpenCLKVCacheOp.hpp"
#include "Types.hpp"
#include "utils/OpenCLTools.hpp"

namespace mllm {

OpenCLKVCacheOp::OpenCLKVCacheOp(Backend *bn, std::string name, int hidden, int head, int n_rep, bool fa2, int cache_max) :
    Op(bn, std::move(name)), hidden_(hidden), head_(head), n_rep_(n_rep), cache_limit_(cache_max), fa2_(fa2) {
    if (fa2_) {
        n_rep_ = 1; // Flash Attention 2 does not use n_rep
    }
    ocl_backend_ = dynamic_cast<OpenCLBackend *>(backend_);
    if (ocl_backend_ == nullptr) throw std::runtime_error("Backend is not OpenCLBackend");

    cache_ = std::make_shared<Tensor>(bn);
    cache_->setName(name + ".Cache");
    const int KVCache_batch = 1;
    cache_->reshape(KVCache_batch, head_ * n_rep_, cache_limit_, hidden_);
    cache_->setDtype(MLLM_TYPE_F32);
    cache_->alloc();
    cache_->cl();

    const std::string kernel_path = "kernel/kvcache.cl";
    cl_program program = ocl_backend_->getProgram(kernel_path);

    cl_int err;
    // Load BSHD kernels
    kernel_fp32_bshd_ = clCreateKernel(program, "update_kv_cache_fp32_bshd", &err);
    check_cl_error(err, "clCreateKernel for update_kv_cache_fp32_bshd");
    kernel_fp16_bshd_ = clCreateKernel(program, "update_kv_cache_fp16_bshd", &err);
    check_cl_error(err, "clCreateKernel for update_kv_cache_fp16_bshd");

    // Load BHSD kernels
    kernel_fp32_bhsd_ = clCreateKernel(program, "update_kv_cache_fp32_bhsd", &err);
    check_cl_error(err, "clCreateKernel for update_kv_cache_fp32_bhsd");
    kernel_fp16_bhsd_ = clCreateKernel(program, "update_kv_cache_fp16_bhsd", &err);
    check_cl_error(err, "clCreateKernel for update_kv_cache_fp16_bhsd");
}

OpenCLKVCacheOp::~OpenCLKVCacheOp() {
    if (kernel_fp32_bshd_) clReleaseKernel(kernel_fp32_bshd_);
    if (kernel_fp16_bshd_) clReleaseKernel(kernel_fp16_bshd_);
    if (kernel_fp32_bhsd_) clReleaseKernel(kernel_fp32_bhsd_);
    if (kernel_fp16_bhsd_) clReleaseKernel(kernel_fp16_bhsd_);
}

ErrorCode OpenCLKVCacheOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->setCtype(inputs[0]->ctype());
    const int new_sequence_length = cache_seq_len_ + inputs[0]->sequence();
    outputs[0]->reshape(cache_->batch(), cache_->head(), new_sequence_length, cache_->dimension());
    outputs[0]->setDtype(cache_->dtype());
    if (inputs[0]->ctype() == BHSD && cache_->ctype() != inputs[0]->ctype()) {
        // cache_->cpu();
        cache_->setCtype(BHSD);
        cache_->reshape(cache_->batch(), head_ * n_rep_, cache_limit_, hidden_);
        // cache_->alloc();
        // cache_->cl();
    }
    if (cache_->dtype() != inputs[0]->dtype()) {
        cache_->setDtype(inputs[0]->dtype());
        cache_->alloc();
    }
    return MLLM_NO_ERROR;
}

ErrorCode OpenCLKVCacheOp::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    inputs[0]->to(MLLM_OPENCL);
    outputs[0]->setDtype(cache_->dtype());
    outputs[0]->alloc();

    return MLLM_NO_ERROR;
}

ErrorCode OpenCLKVCacheOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto output = outputs[0];

    if (cache_seq_len_ + input->sequence() > cache_limit_) {
        std::cerr << "KVCache is full, cannot update." << std::endl;
        return MLLM_NO_ERROR; // Cache is full
    }

    cl_kernel kernel_to_use = nullptr;
    auto ctype = input->ctype();

    if (input->dtype() == MLLM_TYPE_F32) {
        if (cache_->dtype() != MLLM_TYPE_F32) { /* Realloc logic for cache_ */
        }
        if (ctype == BSHD) {
            kernel_to_use = kernel_fp32_bshd_;
        } else if (ctype == BHSD) {
            kernel_to_use = kernel_fp32_bhsd_;
        }
    } else if (input->dtype() == MLLM_TYPE_F16) {
        if (cache_->dtype() != MLLM_TYPE_F16) { /* Realloc logic for cache_ */
        }
        if (ctype == BSHD) {
            kernel_to_use = kernel_fp16_bshd_;
        } else if (ctype == BHSD) {
            kernel_to_use = kernel_fp16_bhsd_;
        }
    }

    cl_mem src_buf = ocl_backend_->get_cl_mem(*input);
    cl_mem cache_buf = ocl_backend_->get_cl_mem(*cache_);

    const int h_in = input->head();
    const int s_in = input->sequence();
    const int d_in = input->dimension();
    const int h_cache = cache_->head();
    const int s_cache = cache_->sequence();

    clSetKernelArg(kernel_to_use, 0, sizeof(cl_mem), &src_buf);
    clSetKernelArg(kernel_to_use, 1, sizeof(cl_mem), &cache_buf);
    clSetKernelArg(kernel_to_use, 2, sizeof(int), &h_in);
    clSetKernelArg(kernel_to_use, 3, sizeof(int), &s_in);
    clSetKernelArg(kernel_to_use, 4, sizeof(int), &d_in);
    clSetKernelArg(kernel_to_use, 5, sizeof(int), &h_cache);
    clSetKernelArg(kernel_to_use, 6, sizeof(int), &s_cache);
    clSetKernelArg(kernel_to_use, 7, sizeof(int), &n_rep_);
    clSetKernelArg(kernel_to_use, 8, sizeof(int), &cache_seq_len_);

    const size_t global_work_size[3] = {(size_t)d_in, (size_t)s_in, (size_t)h_in * n_rep_};

    cl_int err = clEnqueueNDRangeKernel(ocl_backend_->getQueue(), kernel_to_use, 3, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
    check_cl_error(err, "clEnqueueNDRangeKernel for KVCache Update");

    cl_mem out_buf = ocl_backend_->get_cl_mem(*output);
    if (ctype == BHSD) {
        const size_t batch_size = output->batch();
        const size_t head_size = output->head();
        const size_t seq_len_out = output->sequence();
        const size_t dim_size = output->dimension();
        const size_t seq_len_cache = cache_->sequence(); // cache_limit
        const size_t dtype_size = output->dtypeSize();
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t h = 0; h < head_size; ++h) {
                size_t src_offset_bytes = (b * head_size * seq_len_cache + h * seq_len_cache) * dim_size * dtype_size;
                size_t dst_offset_bytes = (b * head_size * seq_len_out + h * seq_len_out) * dim_size * dtype_size;
                size_t bytes_per_head = seq_len_out * dim_size * dtype_size;
                cl_event event;
                err = clEnqueueCopyBuffer(
                    ocl_backend_->getQueue(),
                    cache_buf,
                    out_buf,
                    src_offset_bytes,
                    dst_offset_bytes,
                    bytes_per_head,
                    0, nullptr, &event);
                ocl_backend_->addProfilingEvent(this->name(), event);
                check_cl_error(err, "clEnqueueCopyBuffer for BHSD head");
            }
        }
    } else {
        // 对于 BSHD 布局，可以直接进行线性复制
        size_t bytes_to_copy = output->count() * output->dtypeSize();
        cl_event event;
        err = clEnqueueCopyBuffer(ocl_backend_->getQueue(), cache_buf, out_buf, 0, 0, bytes_to_copy, 0, nullptr, &event);
        ocl_backend_->addProfilingEvent(this->name(), event);
        check_cl_error(err, "clEnqueueCopyBuffer from cache to output");
    }

    cache_seq_len_ += s_in;

    return MLLM_NO_ERROR;
}

} // namespace mllm
