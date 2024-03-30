#ifndef MLLM_QNN_COMMON_OP_H
#define MLLM_QNN_COMMON_OP_H

#include "Op.hpp"
#include "QNNBackend.hpp"
#include "Types.hpp"

namespace mllm {
class QNNCommonOp : public Op {
public:
    QNNCommonOp(Backend *bn, string opName):
        Op(bn, opName) {
    }
    virtual ~QNNCommonOp() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        std::cout << name() <<" reshape"<< std::endl;
        for (auto output : outputs) {
            output->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        }
        return MLLM_NO_ERROR;    
    };
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        std::cout << name() <<" setUp"<< std::endl;
        return MLLM_NO_ERROR;    
    };
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        std::cout << name() <<" execute"<< std::endl;
        return MLLM_NO_ERROR;
    };
    virtual ErrorCode load(AbstructLoader &loader) override {
        return Op::load(loader);
    };
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        return MLLM_NO_ERROR;
    }

protected:
    vector<string *> inputTensorNames_;
    QNNBackend *qnnBackend_;
};

class QNNCommCreator : public QNNBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name) const override {
        return new QNNCommonOp(bn, name);
    }
};
} // namespace mllm

#endif // MLLM_QNN_COMMON_OP_H