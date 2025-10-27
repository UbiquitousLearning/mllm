/**
 * @file XpRoPE.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-10-14
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

class XpRoPE final : public Op, public XpTensorDefineInterface<XpRoPE> {
public:
    XpRoPE(Backend *bk, float rope_theta, int max_position_embeddings, const std::string &op_name, int thread_count) :
        Op(bk, op_name), rope_theta_(rope_theta), max_position_embeddings_(max_position_embeddings), thread_count_(thread_count) {
        sin_params_.setBackend(bk);
        cos_params_.setBackend(bk);
    }

    ~XpRoPE() override = default;

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode load(AbstructLoader &loader) override;

    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    static int input_dims_previous_;
    static Tensor sin_params_;
    static Tensor cos_params_;
    float rope_theta_ = 0.f;
    int max_position_embeddings_ = 0;
    int thread_count_ = 4;
    int h_cnt_ = 0;
};

struct XpRoPECreator : public XnnpackBackend::Creator {
    Op *create(OpParam op_param, Backend *bk, const string &name, int thread_count) const override;
};

} // namespace mllm::xnnpack