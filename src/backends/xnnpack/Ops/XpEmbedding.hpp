/**
 * @file XpEmbedding.hpp
 * @author your name (you@domain.com)
 * @version 0.1
 * @date 2024-10-24
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "Backend.hpp"
#include "Op.hpp"
#include "backends/xnnpack/XnnpackBackend.hpp"
#include "backends/xnnpack/XpInterface.hpp"

namespace mllm::xnnpack {

class XpEmbedding final : public Op, public XpTensorDefineInterface<XpEmbedding> {
public:
    XpEmbedding(Backend *bk, const std::string &op_name, int hidden_size, int vocab_size, int thread_count) :
        Op(bk, op_name), thread_count_(thread_count), hidden_size_(hidden_size), vocab_size_(vocab_size) {
        assert(hidden_size_ > 0);
        assert(vocab_size_ > 0);
        weight_params_.setBackend(backend());
    }

    ~XpEmbedding() override = default;

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode load(AbstructLoader &loader) override;

    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    Tensor weight_params_;
    int hidden_size_ = 0;
    int vocab_size_ = 0;
    int thread_count_ = 4;
};

struct XpEmbeddingCreator : public XnnpackBackend::Creator {
    Op *create(OpParam op_param, Backend *bk, const string &name, int thread_count) const override;
};

} // namespace mllm::xnnpack
