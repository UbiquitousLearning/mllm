/**
 * @file XpLinear.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-10-11
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "Backend.hpp"
#include "Op.hpp"
#include "Tensor.hpp"
#include "backends/xnnpack/XnnpackBackend.hpp"
#include "backends/xnnpack/XpInterface.hpp"

namespace mllm::xnnpack {

class XpLinear final : public Op, public XpTensorDefineInterface<XpLinear> {
public:
    XpLinear(Backend *bk, const std::string &op_name, int in_features, int out_features, bool bias, int thread_count);

    ~XpLinear() override = default;

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode load(AbstructLoader &loader) override;

    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    Tensor bias_params_;
    Tensor weight_params_;
    bool bias_ = true;
    int in_features_ = 0;
    int out_features_ = 0;
    int thread_count_ = 4;
};

struct XpLinearCreator : public XnnpackBackend::Creator {
    Op *create(OpParam op_param, Backend *bk, const string &name, int thread_count) const override;
};
} // namespace mllm::xnnpack