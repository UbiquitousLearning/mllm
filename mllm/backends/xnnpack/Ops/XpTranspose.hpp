/**
 * @file XpTranspose.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
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

class XpTranspose final : public Op, public XpTensorDefineInterface<XpTranspose> {
public:
    XpTranspose(Backend *bk, int axis0, int axis1, const std::string &op_name, int thread_count) :
        Op(bk, op_name), axis0_(axis0), axis1_(axis1), thread_count_(thread_count) {
    }

    ~XpTranspose() override = default;

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int axis0_;
    int axis1_;
    int thread_count_ = 4;
};

struct XpTransposeCreator : public XnnpackBackend::Creator {
    Op *create(OpParam op_param, Backend *bk, const string &name, int thread_count) const override;
};

} // namespace mllm::xnnpack