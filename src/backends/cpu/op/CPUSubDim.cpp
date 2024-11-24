
#include "CPUSubDim.hpp"

namespace mllm {

CPUSubDim::CPUSubDim(Backend *bn,  string opName,Chl dim, vector<int> interval, int threadCount) : thread_count(threadCount),
    Op(bn, opName) {
    dim_ = dim;
    start_d_ = interval[0];
    start_d_const_ = interval[0];
    end_d_ = interval[1];
}

ErrorCode CPUSubDim::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    auto input = inputs[0];
    switch (dim_) {
    case BATCH: {
        if(inputs.size() == 2) {
            outputs[0]->reshape(inputs[1]->batch(), input->head(), input->sequence(), input->dimension());
        }else {
            assert(inputs.size() == 1);
            assert(end_d_ - start_d_ >= 1);
            if(start_d_const_<0) {
                int tmplen = end_d_ - start_d_;
                start_d_ = inputs[0]->batch() + start_d_const_;
                end_d_ = start_d_ + tmplen;
            }
            outputs[0]->reshape(end_d_ - start_d_, input->head(), input->sequence() , input->dimension());
        }
        break;
    }
    case HEAD:{
        if(inputs.size() == 2) {
            outputs[0]->reshape(input->batch(), inputs[1]->head(), input->sequence(), input->dimension());
        }else {
            assert(inputs.size() == 1);
            assert(end_d_ - start_d_ >= 1);
            if(start_d_<0) {
                int tmplen = end_d_ - start_d_;
                start_d_ = inputs[0]->head() + start_d_;
                end_d_ = start_d_ + tmplen;
            }
            outputs[0]->reshape(input->batch(), end_d_ - start_d_, input->sequence() , input->dimension());
        }
        break;
    }
    case SEQUENCE:{
        if(inputs.size() == 2) {
            if(inputs[1]->dimension() == inputs[0]->batch() &&inputs[0]->batch()>1 && inputs[1]->head() == 1 && inputs[1]->sequence() == 1 && inputs[1]->batch() == 1) {
                outputs[0]->reshape(input->batch(), input->head(), 1, input->dimension());
            }else {
                outputs[0]->reshape(input->batch(), input->head(), inputs[1]->sequence(), input->dimension());
            }
        }else {
            assert(inputs.size() == 1);
            assert(end_d_ - start_d_ >= 1);
            if(start_d_const_<0) {
                int tmplen = end_d_ - start_d_;
                start_d_ = inputs[0]->sequence() + start_d_const_;
                end_d_ = start_d_ + tmplen;
            }
            outputs[0]->reshape(input->batch(), input->head(), end_d_ - start_d_ , input->dimension());
        }
        break;
    }
    case DIMENSION:{
        if(inputs.size() == 2) {
            outputs[0]->reshape(input->batch(), input->head(), input->sequence(), inputs[1]->dimension());
        }else {
            assert(inputs.size() == 1);
            assert(end_d_ - start_d_ >= 1);
            if(start_d_<0) {
                int tmplen = end_d_ - start_d_;
                start_d_ = inputs[0]->dimension() + start_d_;
                end_d_ = start_d_ + tmplen;
            }
            outputs[0]->reshape(input->batch(), input->head(), input->sequence() , end_d_ - start_d_);
        }
        break;
    }
    default:
        break;
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUSubDim::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    auto input = inputs[0];
    auto output = outputs[0];
    switch (dim_) {
    case BATCH: {
        std::cout<<"Nor Support"<<std::endl;
        break;
    }
    case HEAD:{
        std::cout<<"Nor Support"<<std::endl;
        break;
    }
    case SEQUENCE:{
        if(inputs.size() == 2) {
            if(inputs[1]->dimension() == inputs[0]->batch() &&inputs[0]->batch()>1 && inputs[1]->head() == 1 && inputs[1]->sequence() == 1 && inputs[1]->batch() == 1) {
                for (int b = 0; b < input->batch(); ++b) {
                    memcpy(output->hostPtr<float>() + output->offset(b, 0, 0, 0),
                           input->hostPtr<float>() + input->offset(b, 0, inputs[1]->dataAt<float>(0, 0, 0, b), 0),
                           input->head() * 1 * input->dimension() * sizeof(float));
                }
            }else {
                for (int b = 0; b < input->batch(); ++b) {
                    memcpy(output->hostPtr<float>() + output->offset(b, 0, 0, 0),
                           input->hostPtr<float>() + input->offset(b, 0, 0, 0),
                           input->head() * inputs[1]->sequence() * input->dimension() * sizeof(float));
                }
            }
        }else {
            for (int b = 0; b < input->batch(); ++b) {
                memcpy(output->hostPtr<float>() + output->offset(b, 0, 0, 0),
                       input->hostPtr<float>() + input->offset(b, 0, start_d_, 0),
                       input->head() * (end_d_-start_d_) * input->dimension() * sizeof(float));
            }
        }
        break;
    }
    case DIMENSION:{
        std::cout<<"Nor Support"<<std::endl;
        break;
    }
        default:
            break;
    }
    return Op::execute(inputs, outputs);
}

} // namespace mllm

