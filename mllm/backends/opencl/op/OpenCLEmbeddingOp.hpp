#ifndef OPENCL_EMBEDDING_OP_HPP
#define OPENCL_EMBEDDING_OP_HPP

#include "Op.hpp"
#include "../OpenCLBackend.hpp"

namespace mllm {

class OpenCLEmbeddingOp : public Op {
public:
    OpenCLEmbeddingOp(Backend *bn, std::string name, int vocab_size, int hidden_size);
    ~OpenCLEmbeddingOp() override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode load(AbstructLoader &loader) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int vocab_size_;
    int hidden_size_;
    Tensor weight_;

    cl_kernel kernel_fp32_ = nullptr;
    cl_kernel kernel_q4_0_ = nullptr;
    cl_kernel kernel_q4_0_fp16_ = nullptr;

    OpenCLBackend *ocl_backend_ = nullptr;
};

class OpenCLEmbeddingOpCreator : public OpenCLBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        int vocab_size = op_param["vocab_size"];
        int hidden_size = op_param["hidden_size"];
        return new OpenCLEmbeddingOp(bn, name, vocab_size, hidden_size);
    }
};

} // namespace mllm

#endif // OPENCL_EMBEDDING_OP_HPP