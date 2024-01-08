
#include "CPUTranspose.hpp"

namespace mllm {

CPUTranspose::CPUTranspose(Backend *bn,  string opName, int threadCount) : thread_count(threadCount),
    Op(bn, opName) {
}

ErrorCode CPUTranspose::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUTranspose  reshape" << std::endl;
    // inputs[0]->transShape(SEQUENCE, DIMENSION);
    if(inputs[0]->ctype() == BSHD) {
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->dimension(), inputs[0]->sequence());
    } else if(inputs[0]->ctype() == BCTHW) {
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->time(), inputs[0]->height(), inputs[0]->width(), inputs[0]->channel());
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUTranspose::load(AbstructLoader &loader) {
    //std::cout<<name() << "  CPUTranspose load" << std::endl;
    return Op::load(loader);
}

ErrorCode CPUTranspose::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUTranspose()" << std::endl;
    return Op::execute(inputs, outputs);
}

ErrorCode CPUTranspose::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUTranspose() free" << std::endl;
    return Op::free(inputs, outputs);
}

ErrorCode CPUTranspose::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUTranspose() setUp" << std::endl;
    // return Op::setUp(inputs, outputs);
    if(inputs[0]->masterTensor() == nullptr) {
        inputs[0]->free();
    }
    outputs[0]->setDtype(activation_dtype());
    outputs[0]->alloc();
    // outputs[0]->transShape(SEQUENCE, DIMENSION);
    inputs[0]->deepCopyFrom(outputs[0].get(), false);
    if(inputs[0]->ctype() == BSHD) {
        inputs[0]->transShape(SEQUENCE, DIMENSION, true);
    }else {
        inputs[0]->transShape(THW, CHANNLE, true);
    }
    // inputs[0]->setUndiffusion();
    return MLLM_NO_ERROR;
}
} // namespace mllm

