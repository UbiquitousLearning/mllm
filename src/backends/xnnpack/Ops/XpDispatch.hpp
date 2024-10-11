/**
 * @file XpDispatch.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-10-06
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "Backend.hpp"
#include "Op.hpp"
#include "backends/xnnpack/XnnpackBackend.hpp"

namespace mllm::xnnpack {
class XpDispatch final : public Op {
public:
    XpDispatch(Backend *bk, const string &name, int thread_count) :
        Op(bk, name), thread_count_(thread_count) {
    }

    ~XpDispatch() override = default;

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    inline void setClearXnnGraphAfterDispatch(bool clear_xnn_graph_after_dispatch) {
        clear_xnn_graph_after_dispatch_ = clear_xnn_graph_after_dispatch;
    }

private:
    int thread_count_;
    bool clear_xnn_graph_after_dispatch_ = true;
};

struct XpDispatchCreator : public XnnpackBackend::Creator {
    Op *create(OpParam op_param, Backend *bk, const string &name, int thread_count) const override;
};

} // namespace mllm::xnnpack
