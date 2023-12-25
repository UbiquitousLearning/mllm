
#include "QNNLinear.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNLinear::QNNLinear(Backend *bn, string opName, int in_features, int out_features, bool bias) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNLinear::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return NO_ERROR;
}

ErrorCode QNNLinear::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return graphAddNode(name(), "MatMul", inputs, outputs);
}

ErrorCode QNNLinear::load(AbstructLoader &loader) {
    // std::cout << name() << "  CPULinear load" << std::endl;
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, out_features_, in_features_);
    weight_.setDtype(loader.getDataType(weight_.name()));
    weight_.alloc();
    loader.load(&weight_);
    if (support_bias_) {
        bias_.setName(name() + ".bias");
        bias_.reshape(1, 1, 1, out_features_);
        bias_.setDtype(loader.getDataType(bias_.name()));
        bias_.alloc();
        loader.load(&bias_);
    }
    return Op::load(loader);
}
} // namespace mllm

