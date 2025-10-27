/**
 * @file XpRMSNorm.hpp
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

class XpRMSNorm final : public Op, public XpTensorDefineInterface<XpRMSNorm> {
public:
    XpRMSNorm(Backend *bk, const std::string &op_name, int norm_size, float epsilon = 1e-6, bool add_unit_offset = false, int thread_count = 4) :
        Op(bk, op_name), norm_size_(norm_size), epsilon_(epsilon), add_unit_offset_(add_unit_offset), thread_count_(thread_count) {
        weight_params_.setBackend(bk);
        epsilon_param_.setBackend(bk);
    }

    ~XpRMSNorm() override = default;

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode load(AbstructLoader &loader) override;

    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    Tensor weight_params_;
    Tensor epsilon_param_;
    float epsilon_;
    int norm_size_;
    bool add_unit_offset_;
    int axis_ = 1;
    int thread_count_ = 4;
};

struct XpRMSNormCreator : public XnnpackBackend::Creator {
    Op *create(OpParam op_param, Backend *bk, const string &name, int thread_count) const override;
};
} // namespace mllm::xnnpack