
#include "CPUParameter.hpp"

namespace mllm {

CPUParameter::CPUParameter(Backend *bn,  string opName,int batch, int head, int seq, int dim, bool multiThread) :
    Op(bn, opName) {
    batch_ = batch;
    head_ = head;
    seq_ = seq;
    dim_ = dim;
    weight_.setBackend(bn);
}

ErrorCode CPUParameter::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUParameter  reshape" << std::endl;
    outputs[0]->reshape(batch_, head_, seq_, dim_);
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUParameter::load(AbstructLoader &loader) {
    //std::cout<<name() << "  CPUParameter load" << std::endl;
    weight_.setName(name());
    weight_.reshape(batch_, head_, seq_, dim_);
    if (&loader != nullptr) {
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        loader.load(&weight_);
    } else {
        weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
    }
    return Op::load(loader);
}

ErrorCode CPUParameter::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUParameter()" << std::endl;
    if(outputs[0]->masterTensor()->name() != weight_.name()) {
        // outputs[0]->copyFrom(weight_);
        for (int n = 0; n < outputs[0]->batch(); ++n) {
            for (int c = 0; c < outputs[0]->head(); ++c) {
                for (int h = 0; h < outputs[0]->sequence(); ++h) {
                    for (int w = 0; w < outputs[0]->dimension(); ++w) {
                        outputs[0]->setDataAt<float>(n, c, h, w, weight_.dataAt<float>(n, c, h, w));
                    }
                }
            }
        }
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPUParameter::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUParameter() free" << std::endl;
    weight_.free();
    return Op::free(inputs, outputs);
}

ErrorCode CPUParameter::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUParameter() setUp" << std::endl;
    outputs[0]->deepCopyFrom(&weight_, false);
    return MLLM_NO_ERROR;
}
} // namespace mllm

