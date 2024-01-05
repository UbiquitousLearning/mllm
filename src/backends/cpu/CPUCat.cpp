
#include "CPUCat.hpp"

namespace mllm {

CPUCat::CPUCat(Backend *bn,  string opName,Chl axis,  bool multiThread) :
    Op(bn, opName) {
    axis_ = axis;
}

ErrorCode CPUCat::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUCat  reshape" << std::endl;
    expd_batch_ = inputs[0]->batch();
    for (int ii = 0; ii < inputs.size(); ++ii) {
        auto input = inputs[ii];
        if(input->batch()>expd_batch_) {
            expd_batch_ = input->batch();
            expd_batch_input_idx = ii;
        }
    }
    switch (axis_) {
    case BATCH: {
        int batch_size=0;
        for (auto input : inputs) {
            batch_size += input->batch();
        }
        outputs[0]->reshape(batch_size, inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        break;
    }
    case HEAD: {
        int head_size=0;
        for (auto input : inputs) {
            head_size += input->head();
        }
        outputs[0]->reshape(expd_batch_, head_size, inputs[0]->sequence(), inputs[0]->dimension());
        break;
    }
    case SEQUENCE: {
        int seq_size=0;
        for (auto input : inputs) {
            seq_size += input->sequence();
        }
        outputs[0]->reshape(expd_batch_, inputs[0]->head(), seq_size, inputs[0]->dimension());
        break;
    }
    case DIMENSION: {
        int dim_size=0;
        for (auto input : inputs) {
            dim_size += input->dimension();
        }
        outputs[0]->reshape(expd_batch_, inputs[0]->head(), inputs[0]->sequence(), dim_size);
        break;
    }
    default:
        break;
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUCat::load(AbstructLoader &loader) {
    //std::cout<<name() << "  CPUCat load" << std::endl;
    return Op::load(loader);
}

ErrorCode CPUCat::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUCat()" << std::endl;
    if(axis_ == DIMENSION) {
        for (int n = 0; n < expd_batch_; ++n) {
            for (int c = 0; c < inputs[0]->head(); ++c) {
                for (int h = 0; h < inputs[0]->sequence(); ++h) {
                    // for (int w = 0; w < inputs[0]->dimension(); ++w) {
                    //     outputs[0]->setDataAt<float>(n, c, h, w, inputs[0]->dataAt<float>(n, c, h, w));
                    // }
                    int w = 0;
                    for (int idx=0; idx<inputs.size(); idx++) {
                        int dim_size=inputs[idx]->dimension();
                        // for (; w < dim_size; ++w) {
                        //     outputs[0]->setDataAt<float>(n, c, h, w, inputs[idx]->dataAt<float>(n, c, h, w));
                        // }
                        auto n_ = n;
                        if(idx != expd_batch_input_idx) {
                            n_ = 0;
                        }
                        memcpy(outputs[0]->ptrAt<float>(n, c, h, w), inputs[idx]->ptrAt<float>(n_, c, h, 0), sizeof(float) * (dim_size));
                        w += dim_size;
                    }
                }
            }
        }
    }
    else if (axis_ == SEQUENCE && inputs[0]->head() == 1 ) {
        for (int n = 0; n < expd_batch_; ++n) {
                // for (int w = 0; w < inputs[0]->dimension(); ++w) {
                    // for (int h = 0; h < inputs[0]->sequence(); ++h) {
                    //     outputs[0]->setDataAt<float>(n, c, h, w, inputs[0]->dataAt<float>(n, c, h, w));
                    // }
                    int h = 0;
                    for (int idx=0; idx<inputs.size(); idx++) {
                        // int seq_size=inputs[idx]->sequence() * inputs[idx]->dimension();
                        // for (; h < seq_size; ++h) {
                        //     outputs[0]->setDataAt<float>(n, c, h, w, inputs[idx]->dataAt<float>(n, c, h, w));
                        // }
                        auto n_ = n;
                        if(idx != expd_batch_input_idx) {
                            n_ = 0;
                        }
                        memcpy(outputs[0]->ptrAt<float>(n, 0, h, 0),
                            inputs[idx]->ptrAt<float>(n_, 0, 0, 0),
                            sizeof(float) * (inputs[idx]->sequence() * inputs[idx]->dimension()));
                        h += inputs[idx]->sequence();
                    }
                // }
            }
        }
    // }
    return Op::execute(inputs, outputs);
}

ErrorCode CPUCat::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUCat() free" << std::endl;
    return Op::free(inputs, outputs);
}

ErrorCode CPUCat::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUCat() setUp" << std::endl;
//     if(axis_ != DIMENSION) {
//         CHECK_EQ(outputs.size(), 1);
//         outputs[0]->setDtype(activation_dtype());
//         outputs[0]->alloc();
//         int cbatch=0;
//         int chead=0;
//         int cseq=0;
//         int cdim=0;
//         for(int idx=0; idx<inputs.size(); idx++) {
//             if (inputs[idx]->masterTensor() ==nullptr) {
//                 inputs[idx]->free();
//             }
//             if(idx >0) {
//                 switch (axis_) {
//                 case BATCH: {
//                     cbatch += inputs[idx-1]->batch();
//                     break;
//                 }
//                 case HEAD: {
//                     chead += inputs[idx-1]->head();
//                     break;
//                 }
//                 case SEQUENCE: {
//                     cseq += inputs[idx-1]->sequence();
//                     break;
//                 }
//                 // case DIMENSION: {
//                 //     cdim += inputs[idx-1]->dimension();
//                 //     break;
//                 // }
//                 default:
//                     break;
//                 }
//             }
//             inputs[idx]->deepCopyFrom(outputs[0].get(), {cbatch, chead, cseq, cdim}); //b,h,s,d
//         }
// #ifdef DEBUG
//         std::cout << "*"<<name()<<" setUp*" << std::endl;
// #endif
//         return MLLM_NO_ERROR;
//     }else {
        return Op::setUp(inputs, outputs);
    // }
}
} // namespace mllm

