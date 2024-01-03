
#include "QNNRMSNorm.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNRMSNorm::QNNRMSNorm(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNRMSNorm::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return NO_ERROR;
}

ErrorCode QNNRMSNorm::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return graphAddNode(name(), "Add", inputs, outputs);
}

ErrorCode QNNRMSNorm::load(AbstructLoader &loader){
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, 1, normSize_); //
    weight_.setDtype(loader.getDataType(weight_.name()));
    weight_.alloc();
    // TEST
    //    weight_.fullData<float>(2.0);
    //    inputs[0]->fullDataTest();
    loader.load(&weight_);
    return Op::load(loader);
}
} // namespace mllm

