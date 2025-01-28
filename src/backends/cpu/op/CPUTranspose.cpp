
#include "CPUTranspose.hpp"

namespace mllm {

CPUTranspose::CPUTranspose(Backend *bn,  string opName, int axis0, int axis1, int threadCount) : thread_count(threadCount),
    Op(bn, opName) {
    axis0_ = (Chl)axis0;
    axis1_ = (Chl)axis1;
}

ErrorCode CPUTranspose::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    // inputs[0]->transShape(SEQUENCE, DIMENSION);
    if(axis0_ == SEQUENCE && axis1_ == DIMENSION) {
        if(inputs[0]->ctype() == BSHD) {
            outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->dimension(), inputs[0]->sequence());
        }
    }
    else if(axis0_ == THW && axis1_ == CHANNLE) {
        if(inputs[0]->ctype() == BCTHW) {
            outputs[0]->reshape(inputs[0]->batch(), inputs[0]->time(), inputs[0]->height(), inputs[0]->width(), inputs[0]->channel());
        }
    }
    else if(axis0_ == BATCH && axis1_ == SEQUENCE) {
        if(inputs[0]->ctype() == BSHD) {
            outputs[0]->reshape(inputs[0]->sequence(), inputs[0]->head(), inputs[0]->batch(), inputs[0]->dimension());
        }
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUTranspose::load(AbstructLoader &loader) {

    return Op::load(loader);
}

ErrorCode CPUTranspose::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    return Op::execute(inputs, outputs);
}

ErrorCode CPUTranspose::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    return Op::free(inputs, outputs);
}

ErrorCode CPUTranspose::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    // return Op::setUp(inputs, outputs);
    if(inputs[0]->masterTensor() == nullptr) {
        inputs[0]->free();
    }
    outputs[0]->setDtype(activation_dtype());
    outputs[0]->alloc();
    // outputs[0]->transShape(SEQUENCE, DIMENSION);
    inputs[0]->shallowCopyFrom(outputs[0].get(), false);
    inputs[0]->transShape(axis0_, axis1_, true);
    // if(inputs[0]->ctype() == BSHD) {
    //     inputs[0]->transShape(SEQUENCE, DIMENSION, true);
    // }else {
    //     inputs[0]->transShape(THW, CHANNLE, true);
    // }
    // inputs[0]->setUndiffusion();
    return MLLM_NO_ERROR;
}
} // namespace mllm

