
#ifndef MLLM_QNNSPLITINPUT_H
#define MLLM_QNNSPLITINPUT_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNSplitInput: public QNNCommonOp {
public:
    QNNSplitInput(Backend *bn, string opName, bool isPrompt, int num);
    virtual ~QNNSplitInput() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    bool isPrompt_;
    int num_;
    Tensor seqs_;
    Tensor scale1_;
    Tensor scale2_;

    Tensor residual_;
};

class QNNSplitInputCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNSplitInput(bn, name, (bool)op_param["isPrompt"], (int)op_param["num"]);
    }
};

} // namespace mllm

#endif
