/**
 * @file XpSoftmax.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-10-16
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

class XpSoftmax final : public Op, public XpTensorDefineInterface<XpSoftmax> {
public:
    XpSoftmax(Backend *bk, int axis, bool do_causal_mask, const std::string &op_name, int thread_count) :
        Op(bk, op_name), axis_(axis), do_causal_mask_(do_causal_mask), thread_count_(thread_count) {
    }

    ~XpSoftmax() override = default;

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int axis_;
    bool do_causal_mask_ = false;
    int thread_count_ = 4;
};

struct XpSoftmaxCreator : public XnnpackBackend::Creator {
    Op *create(OpParam op_param, Backend *bk, const string &name, int thread_count) const override;
};

} // namespace mllm::xnnpack