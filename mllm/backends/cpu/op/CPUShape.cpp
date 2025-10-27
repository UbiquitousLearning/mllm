
#include "CPUShape.hpp"

namespace mllm {

CPUShape::CPUShape(Backend *bn,  string opName,Chl axis, int threadCount) : thread_count(threadCount),
    Op(bn, opName) {
    axis_ = axis;
}

ErrorCode CPUShape::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    int dim = 1;
    if (inputs[0]->ctype() == BTHWC || inputs[0]->ctype() == BCTHW) {
        switch (axis_) {
        case BATCH: {
            dim = inputs[0]->batch();
            break;
        }
        case CHANNLE: {
            dim = inputs[0]->channel();
            break;
        }
        case TIME: {
            dim = inputs[0]->time();
            break;
        }
        case HEIGHT: {
            dim = inputs[0]->height();
            break;
        }
        case WIDTH: {
            dim = inputs[0]->width();
            break;
        }
        default: {
            std::cout << "Unsupport axis type";
        }
        }
    } else {
        switch (axis_) {
        case BATCH: {
            dim = inputs[0]->batch();
            break;
        }
        case HEAD: {
            dim = inputs[0]->head();
            break;
        }
        case SEQUENCE: {
            dim = inputs[0]->sequence();
            break;
        }
        case DIMENSION: {
            dim = inputs[0]->dimension();
            break;
        }
        default: {
            std::cout << "Unsupport axis type";
        }
        }
    }
    outputs[0]->reshape(1,1,dim,1);
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUShape::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    outputs[0]->setDataAt<float>(0,0,0,0, outputs[0]->sequence());
    return Op::execute(inputs, outputs);
}

} // namespace mllm

