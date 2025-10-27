#ifndef OPENCL_KVCACHE_OP_HPP
#define OPENCL_KVCACHE_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLKVCacheOp : public Op {
public:
    OpenCLKVCacheOp(Backend *bn, std::string name, int hidden, int head, int n_rep, bool fa2, int cache_max);
    ~OpenCLKVCacheOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    int getCacheSeqLen() override {
        return cache_seq_len_;
    }
    void clearCache() override {
        cache_seq_len_ = 0;
        cache_->cache_seq_len_ = cache_seq_len_;
    }

private:
    shared_ptr<Tensor> cache_;
    int n_rep_;
    int cache_limit_;
    int cache_seq_len_ = 0;
    int hidden_;
    int head_;
    bool fa2_;

    // Kernels for BSHD layout
    cl_kernel kernel_fp32_bshd_ = nullptr;
    cl_kernel kernel_fp16_bshd_ = nullptr;

    // Kernels for BHSD layout (New)
    cl_kernel kernel_fp32_bhsd_ = nullptr;
    cl_kernel kernel_fp16_bhsd_ = nullptr;

    OpenCLBackend *ocl_backend_ = nullptr;
};

class OpenCLKVCacheOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        int n_rep = (int)op_param["n_rep"];
        int cache_max = (int)op_param["cache_max"];
        bool for_xnn = (bool)op_param["for_xnn"];
        int hidden = (int)op_param["hidden"];
        int head = (int)op_param["head"];
        bool fa2 = (bool)op_param["fa2"];
        return new OpenCLKVCacheOp(bn, name, hidden, head, n_rep, fa2, cache_max);
    }
};

} // namespace mllm

#endif // OPENCL_KVCACHE_OP_HPP