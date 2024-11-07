/**
 * @file XpMatmul.hpp
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

class XpMatMul final : public Op, public XpTensorDefineInterface<XpMatMul> {
public:
    XpMatMul(Backend *bk, bool transpose_a, bool transpose_b, bool transpose_c, const std::string &op_name, int thread_count) :
        Op(bk, op_name), transpose_a_(transpose_a), transpose_b_(transpose_b), transpose_c_(transpose_c), thread_count_(thread_count) {
    }

    ~XpMatMul() override = default;

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    bool transpose_a_ = false;
    bool transpose_b_ = false;
    bool transpose_c_ = false;
    int thread_count_ = 4;
};

struct XpMatMulCreator : public XnnpackBackend::Creator {
    Op *create(OpParam op_param, Backend *bk, const string &name, int thread_count) const override;
};

} // namespace mllm::xnnpack