#include "backends/xnnpack/Ops/XpParameter.hpp"

namespace mllm::xnnpack {

XpParameter::XpParameter(Backend *bn, const string &op_name, int batch, int head, int seq, int dim, int thread_count) :
    thread_count(thread_count),
    Op(bn, op_name) {
    batch_ = batch;
    head_ = head;
    seq_ = seq;
    dim_ = dim;
    weight_.setBackend(bn);
}

ErrorCode XpParameter::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->reshape(batch_, head_, seq_, dim_);
    return Op::reshape(inputs, outputs);
}

ErrorCode XpParameter::load(AbstructLoader &loader) {
    weight_.setName(name());
    weight_.reshape(batch_, head_, seq_, dim_);
    if (loader.getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        loader.load(&weight_);
    } else {
        weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
    }
    return Op::load(loader);
}

ErrorCode XpParameter::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (outputs[0]->masterTensor()->name() != weight_.name()) {
        if (outputs[0]->masterTensor() == nullptr) {
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
        } else {
            if (weight_.batch() == 1) {
                auto off = outputs[0]->shapeOffset();
                auto off_b = off[0];
                auto off_h = off[1];
                auto off_s_ = off[2];
                auto off_d = off[3];
                for (int n = 0; n < outputs[0]->masterTensor()->batch(); ++n) {
                    for (int c = 0; c < outputs[0]->head(); ++c) {
                        for (int h = 0; h < outputs[0]->sequence(); ++h) {
                            for (int w = 0; w < outputs[0]->dimension(); ++w) {
                                outputs[0]->masterTensor()->setDataAt<float>(n + off_b, c + off_h, h + off_s_, w + off_d, weight_.dataAt<float>(0, c, h, w));
                            }
                        }
                    }
                }
            }
        }
    }
    return Op::execute(inputs, outputs);
}

ErrorCode XpParameter::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    return Op::free(inputs, outputs);
}

ErrorCode XpParameter::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->deepCopyFrom(&weight_, false);
    return MLLM_NO_ERROR;
}

Op *XpParameterCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    int batch = (int)op_param["batch"];
    int head = (int)op_param["head"];
    int seq = (int)op_param["seq"];
    int dim = (int)op_param["dim"];
    return new XpParameter(bk, name, batch, head, seq, dim, thread_count);
}
} // namespace mllm::xnnpack