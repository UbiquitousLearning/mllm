

#include "CPUView.hpp"

namespace mllm {

CPUView::CPUView(Backend *bn,  string opName,vector<int> dims, vector<int>data_dims, bool multiThread) :
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
    //std::cout<<name() << "  CPUView  reshape" << std::endl;
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
        }else if (data_dim0_ == SEQUENCE && data_dim1_ == HEAD && data_dim2_ == BATCH && data_dim3_ ==DIMENSION) {
            dim0 = inputs[0]->sequence();
            dim1 = inputs[0]->head();
            dim2 = inputs[0]->batch();
            dim3 = inputs[0]->dimension();
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
    //std::cout<<name() << "  CPUView()" << std::endl;
    /*
    if(data_dim0_ == 0 && data_dim1_ == 1 && data_dim2_ == 2 && data_dim3_ == 3) {
        return Op::execute(inputs, outputs);
        // outputs[0]->copyFrom(inputs[0]);
    } else if(data_dim0_ == 0 && data_dim1_ == 3 && data_dim2_ == 2 && data_dim3_ == 3){
        // if (inputs[0]->ctype() == BSHD && outputs[0]->ctype() == BSHD) {
        // } else
        {
            for (int n = 0; n < inputs[0]->batch(); ++n) {
#pragma omp parallel for num_threads(4)
                for (int h = 0; h < dim1_; ++h) {
                    for (int s = 0; s < inputs[0]->sequence(); ++s) {
                        float* src = inputs[0]->ptrAt<float>(n, 0, s, h * (inputs[0]->dimension() / dim1_));
                        float* dest = outputs[0]->ptrAt<float>(n, h, s, 0);
                        memcpy(dest, src, sizeof(float) * (inputs[0]->dimension() / dim1_));
                    }
                }
            }
        }
    } else if(data_dim0_ == 0 && data_dim1_ == -1 && data_dim2_ == 2 && data_dim3_ == 1+3){
        // if (inputs[0]->ctype() == BSHD && outputs[0]->ctype() == BSHD) {
        // } else
        {
            int batch_size = inputs[0]->batch();
            int head_num = inputs[0]->head();
            int sequence = inputs[0]->sequence();
            int dimension = inputs[0]->dimension();
            for (int n = 0; n < batch_size; ++n) {
                for (int s = 0; s < sequence; ++s) {
#pragma omp parallel for num_threads(4)
                    for (int d = 0; d < dimension; ++d) {
                        for (int h = 0; h < head_num; ++h) {
                            float value = inputs[0]->dataAt<float>(n, h, s, d);
                            float* dest_ptr = outputs[0]->hostPtr<float>() + (n * sequence * head_num * dimension) + (s * head_num * dimension) + (h * dimension) + d;
                            memcpy(dest_ptr, &value, sizeof(float));
                        }
                    }
                }
            }
        }
    }
    */
    return Op::execute(inputs, outputs);
}

ErrorCode CPUView::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    if (   (data_dim0_ == BATCH && data_dim2_ ==SEQUENCE && inputs[0]->ctype()!=BCTHW)  // head & dimension
        || (data_dim0_ == BATCH && data_dim3_ ==DIMENSION && inputs[0]->ctype()==BSHD) // head & sequence
        || (data_dim0_ == SEQUENCE && data_dim1_ == HEAD && data_dim2_ == BATCH && data_dim3_ ==DIMENSION && inputs[0]->ctype()==BSHD) // head & sequence
        || (data_dim0_ == BATCH && inputs[0]->ctype()==BCTHW) //
        // || (data_dim0_ == BATCH && data_dim3_ == CHANNLE && inputs[0]->ctype()==BTHWC) //
    ){
        noNeedEx_ = true;
        if(inputs[0]->masterTensor() == nullptr) {
            inputs[0]->free();
        }
        outputs[0]->setDtype(activation_dtype());
        outputs[0]->alloc();
        inputs[0]->deepCopyFrom(outputs[0].get(), false);
#ifdef DEBUG
        std::cout << "*"<<name()<<" setUp*" << std::endl;
#endif
        return MLLM_NO_ERROR;
    }
    else {
        std::cout<<"CPUView not support!!!!"<<std::endl;
        return Op::setUp(inputs, outputs);
    }
}

} // namespace mllm

