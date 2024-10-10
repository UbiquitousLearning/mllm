/**
 * @file XpDirect.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-10-09
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

class XpDirect final : public Op, public XpTensorDefineInterface<XpDirect> {
public:
    XpDirect(Backend *bk, const std::string &op_name, int thread_count) :
        Op(bk, op_name), thread_count_(thread_count) {
    }

    ~XpDirect() override = default;

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    void setType(XpTensorType type);

private:
    XpTensorType type_ = XpTensorType::Normal;
    int thread_count_ = 4;
};

struct XpDirectCreator : public XnnpackBackend::Creator {
    Op *create(OpParam op_param, Backend *bk, const string &name, int thread_count) const override;
};

} // namespace mllm::xnnpack