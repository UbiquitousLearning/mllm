/**
 * @file XpSDPA.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-10-19
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

class XpSDPA final : public Op, public XpTensorDefineInterface<XpSDPA> {
public:
    XpSDPA(Backend *bk, const std::string &op_name, int thread_count) :
        Op(bk, op_name), thread_count_(thread_count) {
        scale_params_.setBackend(backend());
        mask_params_.setBackend(backend());
    }

    ~XpSDPA() override = default;

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode load(AbstructLoader &loader) override;

    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    Tensor scale_params_;
    Tensor mask_params_;
    int thread_count_ = 4;
};

struct XpSDPACreator : public XnnpackBackend::Creator {
    Op *create(OpParam op_param, Backend *bk, const string &name, int thread_count) const override;
};

} // namespace mllm::xnnpack