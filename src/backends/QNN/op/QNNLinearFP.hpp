
#ifndef MLLM_QNNLINEARFP_H
#define MLLM_QNNLINEARFP_H

#include "QNNCommonOp.hpp"

namespace mllm {
class QNNLinearFP : public QNNCommonOp {
public:
    QNNLinearFP(Backend *bn, string opName, int in_features, int out_features, bool bias);
    virtual ~QNNLinearFP() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int in_features_;
    int out_features_;
    bool support_bias_;
    Tensor weight_;
    Tensor bias_;
};

class QNNLinearFPCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        int in_features = op_param["in_features"];
        int out_features = op_param["out_features"];
        int bias = op_param["bias"];
        return new QNNLinearFP(bn, name, in_features, out_features, (bool)bias);
    }
};

} // namespace mllm

#endif
