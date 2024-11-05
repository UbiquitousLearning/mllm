/**
 * @file XpParameter.hpp
 * @author your name (you@domain.com)
 * @version 0.1
 * @date 2024-10-26
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "Op.hpp"
#include "backends/xnnpack/XnnpackBackend.hpp"
#include "backends/xnnpack/XpInterface.hpp"

namespace mllm::xnnpack {

class XpParameter final : public Op, public XpTensorDefineInterface<XpParameter> {
public:
    XpParameter(Backend *bn, const string &op_name, int batch, int head, int seq, int dim, int thread_count);
    ~XpParameter() override = default;
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode load(AbstructLoader &loader) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    Tensor &weight() {
        auto xpb = (XnnpackBackend *)backend();
        defineWeightTensor(xpb->getCurProcessingGraph(), &weight_);
        return weight_;
    }

private:
    int thread_count = 4;
    Tensor weight_;
    int batch_;
    int head_;
    int seq_;
    int dim_;
};

class XpParameterCreator : public XnnpackBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bk, const string &name, int thread_count) const override;
};

} // namespace mllm::xnnpack