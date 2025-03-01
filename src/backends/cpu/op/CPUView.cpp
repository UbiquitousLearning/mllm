

#include "CPUView.hpp"

namespace mllm {

CPUView::CPUView(Backend *bn,  string opName,vector<int> dims, vector<int>data_dims, int threadCount) : thread_count(threadCount),
    Op(bn, opName) {
    dim0_ = dims[0];
    dim1_ = dims[1];
    dim2_ = dims[2];
    dim3_ = dims[3];
    // if(dims.size() == 5) {
    //     dim4_ = dims[4];
    // }
    data_dim0_ = data_dims[0];
    data_dim1_ = data_dims[1];
    data_dim2_ = data_dims[2];
    data_dim3_ = data_dims[3];
    // if(data_dims.size() == 5) {
    //     data_dim4_ = data_dims[4];
    // }
}

ErrorCode CPUView::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    // if(data_dim4_ != -999) {
    //     int dim0 = inputs[0]->batch();
    //     int dim1 = inputs[0]->channel();
    //     int dim2 = inputs[0]->height();
    //     int dim3 = inputs[0]->width();
    //     int dim4 = inputs[0]->dimension();
    //     assert(inputs[0]->ctype() == BCTHW);
    //
    //     outputs[0]->reshape(dim0, dim1, dim2, dim3, dim4);
    // } else {
        int dim0 = inputs[0]->batch();
        int dim1 = inputs[0]->head();
        int dim2 = inputs[0]->sequence();
        int dim3 = inputs[0]->dimension();
        if(data_dim0_ == BATCH && data_dim1_ == DIMENSION && data_dim2_ == SEQUENCE && data_dim3_ == DIMENSION) {
            dim1 = dim1_;
            dim3 = inputs[0]->dimension()/ dim1_;
        } else if(data_dim0_ == BATCH && data_dim1_ == -1 && data_dim2_ == SEQUENCE && data_dim3_ == HEAD+DIMENSION){
            dim1 = 1;
            dim3 = inputs[0]->dimension() * inputs[0]->head();
        } else if(data_dim0_ == BATCH && data_dim1_ == -1 && data_dim2_ == SEQUENCE+HEAD && data_dim3_ == DIMENSION){
            dim1 = 1;
            dim2 = inputs[0]->sequence()* inputs[0]->head();
        } else if (data_dim0_ == BATCH && data_dim1_ == -1 && data_dim2_ == CHANNLE && data_dim3_ == TIME + HEIGHT + WIDTH) {
            // assert(inputs[0]->ctype() == BCTHW);
            dim1 = 1;
            dim2 = inputs[0]->channel();
            dim3 = inputs[0]->time() * inputs[0]->height() * inputs[0]->width();
        } else if (data_dim0_ == BATCH && data_dim1_ == -1 && data_dim2_ == TIME + HEIGHT + WIDTH && data_dim3_ == CHANNLE ) {
            if(inputs[0]->ctype() == BTHWC) {
                dim1 = 1;
                dim2 = inputs[0]->time() * inputs[0]->height() * inputs[0]->width();
                dim3 = inputs[0]->channel();
            }else {
                dim1 = 1;
                dim2 = inputs[0]->time() * inputs[0]->height() * inputs[0]->channel();
                dim3 = inputs[0]->width();
            }
        } else if (data_dim0_ == SEQUENCE && data_dim1_ == HEAD && data_dim2_ == BATCH && data_dim3_ ==DIMENSION) {
            dim0 = inputs[0]->sequence();
            dim1 = inputs[0]->head();
            dim2 = inputs[0]->batch();
            dim3 = inputs[0]->dimension();
        } else if (data_dim0_ == BATCH && data_dim1_ == HEAD && data_dim2_ == BATCH && data_dim3_ ==DIMENSION) {
            dim0 = inputs[0]->batch()/dim2_;
            dim1 = inputs[0]->head();
            dim2 = dim2_;
            dim3 = inputs[0]->dimension();
        } else if (data_dim0_ == BATCH && data_dim1_ == HEAD && data_dim2_ == SEQUENCE && data_dim3_ ==DIMENSION) {
            dim0 = dim0_;
            dim1 = dim1_;
            dim2 = dim2_;
            dim3 = dim3_;
        } else {
            std::cout<<"CPUView not support!!!!"<<std::endl;
        }
        outputs[0]->reshape(dim0, dim1, dim2, dim3);
    // }
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUView::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if(noNeedEx_){
        return Op::execute(inputs, outputs);
    } else {
        std::cout<<"CPUView not support!!!!"<<std::endl;
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPUView::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);

    activation_dtype_ = inputs[0]->dtype();
    
    if (   (data_dim0_ == BATCH && data_dim2_ ==SEQUENCE && inputs[0]->ctype()!=BCTHW)  // head & dimension
        || (data_dim0_ == BATCH && data_dim3_ ==DIMENSION && inputs[0]->ctype()==BSHD) // head & sequence
        || (data_dim0_ == SEQUENCE && data_dim1_ == HEAD && data_dim2_ == BATCH && data_dim3_ ==DIMENSION && inputs[0]->ctype()==BSHD) // head & sequence
        || (data_dim0_ == BATCH && inputs[0]->ctype()==BCTHW) //
        || (data_dim1_ == HEAD && data_dim3_ ==DIMENSION && inputs[0]->ctype()==BSHD // batch & sequence
        || (data_dim0_ == BATCH && data_dim1_ == HEAD && data_dim2_ == SEQUENCE && data_dim3_ ==DIMENSION)) // batch & sequence & head & dimension
        // || (data_dim0_ == BATCH && data_dim3_ == CHANNLE && inputs[0]->ctype()==BTHWC) //
    ){
        noNeedEx_ = true;
        if (inputs[0]->masterTensor() == nullptr) {
            inputs[0]->free();
        }
        outputs[0]->setDtype(activation_dtype());
        outputs[0]->alloc();
        inputs[0]->shallowCopyFrom(outputs[0].get(), false);
        return MLLM_NO_ERROR;
    }
    else {
        std::cout<<"CPUView not support!!!!"<<std::endl;
        return Op::setUp(inputs, outputs);
    }
}

} // namespace mllm

