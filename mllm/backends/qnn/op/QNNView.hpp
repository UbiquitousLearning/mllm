
#ifndef MLLM_QNNVIEW_H
#define MLLM_QNNVIEW_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNView : public QNNCommonOp {
public:
    QNNView(Backend *bn, string opName, vector<int> dims, vector<int> data_dims);
    virtual ~QNNView() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;

private:
    int dim0_ = -1;
    int dim1_ = -1;
    int dim2_ = -1;
    int dim3_ = -1;
    int data_dim0_;
    int data_dim1_;
    int data_dim2_;
    int data_dim3_;
};

class QNNViewCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        vector<int> dims = {(int)op_param["dim0"], (int)op_param["dim1"], (int)op_param["dim2"], (int)op_param["dim3"]};
        vector<int> data_dims = {(int)op_param["data_dim0"], (int)op_param["data_dim1"], (int)op_param["data_dim2"], (int)op_param["data_dim3"]};
        return new QNNView(bn, name, dims, data_dims);
    }
};

} // namespace mllm

#endif
