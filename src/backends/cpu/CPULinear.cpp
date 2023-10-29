
#include "CPULinear.hpp"

namespace mllm {

CPULinear::CPULinear(Backend *bn, string opName, int in_features, int out_features, bool bias, bool multiThread) :
    Op(bn, opName) {
    in_features_ = in_features;
    out_features_ = out_features;
    support_bias_ = bias;
    support_multi_thread_ = multiThread;
    weight_.setBackend(bn);
    bias_.setBackend(bn);
}

ErrorCode CPULinear::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << name() << "  CPULinear  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    // N     |    C       |   H                   |  W
    // -----------------------------------------------
    // 1     |out_channel | in_channel            |  1
    //       |out_features| in_features           |
    // -----------------------------------------------
    // batch |in_channel  | seq_len               |  1
    //       |in_features | inputs[0]->sequence()   |
    // -----------------------------------------------
    // batch |out_channel | seq_len               |  1
    //       |out_features|  inputs[0]->sequence()  |
    CHECK_EQ(in_features_, inputs[0]->dimension());
    weight_.reshape(1, inputs[0]->head(), out_features_, in_features_);
    weight_.setName(name() + ".weight");
    weight_.setDtype(weightsDtype());
    if (support_bias_) {
        bias_.reshape(1, inputs[0]->head(), 1, out_features_);
        bias_.setName(name() + ".bias");
        bias_.setDtype(weightsDtype());
    }
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), out_features_);
    outputs[0]->setDtype(activationDtype());
    return NO_ERROR;
}

ErrorCode CPULinear::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << name() << "  CPULinear  setUp" << std::endl;
    if (!inputs[0]->allocted()) {
        inputs[0]->alloc(); // TODO remove
    }
    outputs[0]->alloc();
    weight_.alloc();
    //    weight_.fullData<float>(1);
    if (support_bias_) {
        bias_.alloc();
    }
    return NO_ERROR;
}

ErrorCode CPULinear::load(ParamLoader &loader) {
    std::cout << name() << "  CPULinear load" << std::endl;
    loader.load(&weight_);
    if (support_bias_)
        loader.load(&bias_);
    // weight_.printData<float>();
    return NO_ERROR;
}
ErrorCode CPULinear::reshapeOutputs(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << name() << "  CPULinear  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    // N     |    C       |   H                   |  W
    // -----------------------------------------------
    // 1     |out_channel | in_channel            |  1
    //       |out_features| in_features           |
    // -----------------------------------------------
    // batch |in_channel  | seq_len               |  1
    //       |in_features | inputs[0]->sequence()   |
    // -----------------------------------------------
    // batch |out_channel | seq_len               |  1
    //       |out_features|  inputs[0]->sequence()  |
    CHECK_EQ(in_features_, inputs[0]->dimension());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), out_features_);
    outputs[0]->alloc();
    return NO_ERROR;
}

ErrorCode CPULinear::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << name() << "  CPULinear()" << std::endl;
    switch (weightsDtype()) {
    case MLLM_TYPE_F32: {
        mat_mul_fp32(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_, false, true);
        break;
    }
    case MLLM_TYPE_F16: break;
    case MLLM_TYPE_Q4_0: {
        mat_mul_fp32_q4_0(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_, false, true);
        break;
    }
    case MLLM_TYPE_Q4_K: {
        mat_mul_fp32_q4_K(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_, false, true);
        break;
    }
    default:
        break;
    }
    return NO_ERROR;
}
ErrorCode CPULinear::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    if (support_bias_) {
        bias_.free();
    }
    return Op::free(inputs, outputs);
}

} // namespace mllm
