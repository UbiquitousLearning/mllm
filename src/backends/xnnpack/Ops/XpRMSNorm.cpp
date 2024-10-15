#include "backends/xnnpack/Ops/XpRMSNorm.hpp"
#include "backends/xnnpack/XnnpackBackend.hpp"
#include "Types.hpp"

namespace mllm::xnnpack {

ErrorCode XpRMSNorm::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // do nothing
    return MLLM_NO_ERROR;
}

ErrorCode XpRMSNorm::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)backend();
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    defineWeightTensor(xpb, &weight_);
    return Op::reshape(inputs, outputs);
}

ErrorCode XpRMSNorm::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)backend();
    tryDefineAllXpTensors(xpb, inputs);
    tryDefineAllXpTensors(xpb, outputs);

    // TODO

    return MLLM_NO_ERROR;
}

ErrorCode XpRMSNorm::load(AbstructLoader &loader) {
    auto xpb = (XnnpackBackend *)backend();

    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, 1, norm_size_);
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

ErrorCode XpRMSNorm::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    return Op::free(inputs, outputs);
}

Op *XpRMSNormCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    int normSize = (int)op_param["norm_size"];
    float epsilon = (float)op_param["epsilon"];
    bool add_unit_offset_ = (op_param.find("add_unit_offset") == op_param.end()) ? false : true;
    return new XpRMSNorm(bk, name, normSize, epsilon, add_unit_offset_, thread_count);
}
} // namespace mllm::xnnpack