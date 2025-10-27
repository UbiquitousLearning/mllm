
#include "QNNView.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cassert>

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
    int dim0 = inputs[0]->batch();
    int dim1 = inputs[0]->sequence();
    int dim2 = inputs[0]->head();
    int dim3 = inputs[0]->dimension();

    if (data_dim0_ == BATCH && data_dim1_ == DIMENSION && data_dim2_ == SEQUENCE && data_dim3_ == DIMENSION) {
        dim1 = dim1_;
        dim2 = inputs[0]->sequence();
        dim3 = inputs[0]->dimension() / dim1_;
    } else if (data_dim0_ == BATCH && data_dim1_ == -1 && data_dim2_ == SEQUENCE && data_dim3_ == HEAD + DIMENSION) {
        dim2 = dim1;
        dim1 = 1;
        dim3 = inputs[0]->dimension() * inputs[0]->head();
    } else if (data_dim0_ == BATCH && data_dim1_ == -1 && data_dim2_ == SEQUENCE + HEAD && data_dim3_ == DIMENSION) {
        dim1 = 1;
        dim2 = inputs[0]->sequence() * inputs[0]->head();
    } else if (data_dim0_ == BATCH && data_dim1_ == -1 && data_dim2_ == CHANNLE && data_dim3_ == TIME + HEIGHT + WIDTH) {
        // assert(inputs[0]->ctype() == BCTHW);
        dim1 = 1;
        dim2 = inputs[0]->channel();
        dim3 = inputs[0]->time() * inputs[0]->height() * inputs[0]->width();
    } else if (data_dim0_ == BATCH && data_dim1_ == -1 && data_dim2_ == TIME + HEIGHT + WIDTH && data_dim3_ == CHANNLE) {
        if (inputs[0]->ctype() == BTHWC) {
            dim1 = 1;
            dim2 = inputs[0]->time() * inputs[0]->height() * inputs[0]->width();
            dim3 = inputs[0]->channel();
        } else {
            dim1 = 1;
            dim2 = inputs[0]->time() * inputs[0]->height() * inputs[0]->channel();
            dim3 = inputs[0]->width();
        }
    } else if (data_dim0_ == SEQUENCE && data_dim1_ == HEAD && data_dim2_ == BATCH && data_dim3_ == DIMENSION) {
        dim0 = inputs[0]->sequence();
        dim1 = inputs[0]->head();
        dim2 = inputs[0]->batch();
        dim3 = inputs[0]->dimension();
    } else if (data_dim0_ == BATCH && data_dim1_ == HEAD && data_dim2_ == BATCH && data_dim3_ == DIMENSION) {
        dim0 = inputs[0]->batch() / dim2_;
        dim1 = inputs[0]->head();
        dim2 = dim2_;
        dim3 = inputs[0]->dimension();
    } else if (data_dim0_ == BATCH && data_dim1_ == SEQUENCE && data_dim2_ == SEQUENCE && data_dim3_ == DIMENSION) {
        dim0 = inputs[0]->batch();
        dim1 = dim1_;
        dim2 = dim1_;
        dim3 = inputs[0]->dimension();
    } else if (data_dim0_ == BATCH && data_dim1_ == HEAD && data_dim2_ == SEQUENCE && data_dim3_ == DIMENSION) {
        dim0 = dim0_;
        dim1 = dim1_;
        dim2 = dim2_;
        dim3 = dim3_;
    } else {
        std::cout << "QNNView not support!!!!" << std::endl;
    }
    outputs[0]->reshape(dim0, dim1, dim2, dim3);

    return Op::reshape(inputs, outputs);
}

ErrorCode QNNView::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->setDtype(inputs[0]->dtype());

    if (outputs[0]->dtype() == MLLM_TYPE_I8 || outputs[0]->dtype() == MLLM_TYPE_I16) {
        outputs[0]->quant_param = inputs[0]->quant_param;
    }
    return graphAddNode(name(), "Reshape", inputs, outputs, {}, "qti.aisw", true);
}

ErrorCode QNNView::load(AbstructLoader &loader) {
    return Op::load(loader);
}

} // namespace mllm
