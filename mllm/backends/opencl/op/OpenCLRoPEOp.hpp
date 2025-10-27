#ifndef OPENCL_ROPE_OP_HPP
#define OPENCL_ROPE_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"
#include "Types.hpp"

namespace mllm {

class OpenCLRoPEOp : public Op {
public:
    OpenCLRoPEOp(Backend *bn, string opName, OpParam &config, int threadCount);
    OpenCLRoPEOp(Backend *bn, string opName, int pose_type, int threadCount);
    OpenCLRoPEOp(Backend *bn, string opName, int pose_type, float rope_theta, int max_position_embeddings, int threadCount);
    OpenCLRoPEOp(Backend *bn, string opName, int pose_type, float rope_theta, float partial_rotary_factor, int max_position_embeddings, int threadCount);

    ~OpenCLRoPEOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    void clearCache() override {
        pos_offset_ = 0;
    }

private:
    // 原来的成员变量改为 static（全局共享）
    static cl_mem sin_buffer_fp32_;
    static cl_mem cos_buffer_fp32_;
    static cl_mem sin_buffer_fp16_;
    static cl_mem cos_buffer_fp16_;
    static size_t buffer_size_fp32_;
    static size_t buffer_size_fp16_;
    static vector<vector<float>> sin_table_cpu_fp32_;
    static vector<vector<float>> cos_table_cpu_fp32_;
    static int partial_dim_cached_;

    void _init(int threadCount);
    void _computeSinCosTable(int partial_dim);

    // FP32 buffers
    // cl_mem sin_buffer_fp32_ = nullptr;
    // cl_mem cos_buffer_fp32_ = nullptr;
    // size_t buffer_size_fp32_ = 0;
    // vector<vector<float>> sin_table_cpu_fp32_;
    // vector<vector<float>> cos_table_cpu_fp32_;

    // FP16 buffers
    // cl_mem sin_buffer_fp16_ = nullptr;
    // cl_mem cos_buffer_fp16_ = nullptr;
    // size_t buffer_size_fp16_ = 0;

    // FP32 kernels
    cl_kernel kernel_llama_fp32_ = nullptr;
    cl_kernel kernel_hf_fp32_ = nullptr;

    // FP16 kernels
    cl_kernel kernel_llama_fp16_ = nullptr;
    cl_kernel kernel_hf_fp16_ = nullptr;

    OpParam config_;
    RoPEType pose_type_;
    int pos_max_ = 4096;
    float rope_theta_ = 10000.0f;
    float partial_rotary_factor_ = 1.0f;
    // int partial_dim_cached_ = 0;
    int pos_offset_ = 0;

    OpenCLBackend *ocl_backend_ = nullptr;
};

class OpenCLRoPEOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override;
};

} // namespace mllm

#endif // OPENCL_ROPE_OP_HPP