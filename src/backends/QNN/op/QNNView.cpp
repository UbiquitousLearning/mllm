
#include "QNNView.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNView::QNNView(Backend *bn, string opName, vector<int> dims, vector<int> data_dims) :
    QNNCommonOp(bn, opName) {
    dim0_ = dims[0];
    dim1_ = dims[1];
    dim2_ = dims[2];
    dim3_ = dims[3];
    data_dim0_ = data_dims[0];
    data_dim1_ = data_dims[1];
    data_dim2_ = data_dims[2];
    data_dim3_ = data_dims[3];
}

ErrorCode QNNView::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    int dim0;
    int dim1;
    int dim2;
    int dim3;

    if (data_dim0_ == 0 && data_dim1_ == 1 && data_dim2_ == 2 && data_dim3_ == 3) {
        dim0 = inputs[0]->batch();
        dim1 = inputs[0]->head();
        dim2 = inputs[0]->sequence();
        dim3 = inputs[0]->dimension();
    } else if (data_dim0_ == 0 && data_dim1_ == 3 && data_dim2_ == 2 && data_dim3_ == 3) {
        dim0 = inputs[0]->batch();
        dim1 = dim1_;
        dim2 = inputs[0]->sequence();
        dim3 = inputs[0]->dimension() / dim1_;
    } else if (data_dim0_ == 0 && data_dim1_ == -1 && data_dim2_ == 2 && data_dim3_ == 1 + 3) {
        dim0 = inputs[0]->batch();
        dim1 = 1;
        dim2 = inputs[0]->sequence();
        dim3 = inputs[0]->dimension() * inputs[0]->head();
    }
    outputs[0]->reshape(dim0, dim1, dim2, dim3);
    return NO_ERROR;
}

ErrorCode QNNView::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return graphAddNode(name(), "Reshape", inputs, outputs);
}
} // namespace mllm
