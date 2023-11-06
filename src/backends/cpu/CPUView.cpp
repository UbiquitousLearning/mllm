

#include "CPUView.hpp"

namespace mllm {

CPUView::CPUView(Backend *bn,  string opName,vector<int> dims, vector<int>data_dims, bool multiThread) :
    Op(bn, opName) {
    dim0_ = dims[0];
    dim1_ = dims[1];
    dim2_ = dims[2];
    dim3_ = dims[3];
    data_dim0_ = data_dims[0];
    data_dim1_ = data_dims[1];
    data_dim2_ = data_dims[2];
    data_dim3_ = data_dims[3];
}

ErrorCode CPUView::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    int dim0;
    int dim1;
    int dim2;
    int dim3;
    std::cout<<name() << "  CPUView  reshape" << std::endl;
    if(data_dim0_ == 0 && data_dim1_ == 1 && data_dim2_ == 2 && data_dim3_ == 3) {
        dim0 = inputs[0]->batch();
        dim1 = inputs[0]->head();
        dim2 = inputs[0]->sequence();
        dim3 = inputs[0]->dimension();
    } else if(data_dim0_ == 0 && data_dim1_ == 3 && data_dim2_ == 2 && data_dim3_ == 3) {
        dim0 = inputs[0]->batch();
        dim1 = dim1_;
        dim2 = inputs[0]->sequence();
        dim3 = inputs[0]->dimension()/ dim1_;
    } else if(data_dim0_ == 0 && data_dim1_ == -1 && data_dim2_ == 2 && data_dim3_ == 1+3){
        dim0 = inputs[0]->batch();
        dim1 = 1;
        dim2 = inputs[0]->sequence();
        dim3 = inputs[0]->dimension() * inputs[0]->head();
    }
    outputs[0]->reshape(dim0, dim1, dim2, dim3);
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUView::load(ParamLoader &loader) {
    std::cout<<name() << "  CPUView load" << std::endl;
    return NO_ERROR;
}

ErrorCode CPUView::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout<<name() << "  CPUView()" << std::endl;
    if(data_dim0_ == 0 && data_dim1_ == 1 && data_dim2_ == 2 && data_dim3_ == 3) {
        outputs[0]->copyFrom(inputs[0]);
    } else if(data_dim0_ == 0 && data_dim1_ == 3 && data_dim2_ == 2 && data_dim3_ == 3){
        for (int n = 0; n < inputs[0]->batch(); ++n) {
            for (int h = 0; h < dim1_; ++h) {
                #pragma omp parallel for num_threads(8)
                for (int s = 0; s < inputs[0]->sequence(); ++s) {
                    float* src = inputs[0]->ptrAt<float>(n, 0, s, h * (inputs[0]->dimension() / dim1_));
                    float* dest = outputs[0]->ptrAt<float>(n, h, s, 0);
                    memcpy(dest, src, sizeof(float) * (inputs[0]->dimension() / dim1_));
                }
            }
        }
    } else if(data_dim0_ == 0 && data_dim1_ == -1 && data_dim2_ == 2 && data_dim3_ == 1+3){
        int batch_size = inputs[0]->batch();
        int head_num = inputs[0]->head();
        int sequence = inputs[0]->sequence();
        int dimension = inputs[0]->dimension();
        for (int n = 0; n < batch_size; ++n) {
            for (int s = 0; s < sequence; ++s) {
                #pragma omp parallel for num_threads(8)
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
    return NO_ERROR;
}

} // namespace mllm

