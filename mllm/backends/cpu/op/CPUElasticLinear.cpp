
#include "CPUElasticLinear.hpp"
#include "../compute/MatmulElastic.hpp"

namespace mllm {

CPUElasticLinear::CPUElasticLinear(Backend *bn, string opName, int in_features, int out_features, bool bias, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    in_features_ = in_features;
    out_features_ = out_features;
    support_bias_ = bias;
    thread_count = threadCount;
    weight_.setBackend(bn);
    bias_.setBackend(bn);
}

ErrorCode CPUElasticLinear::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << name() << "  CPUElasticLinear  reshape" << std::endl;
    assert(inputs.size() == 3);
    assert(outputs.size() == 1);
    int activate_input_dim = (int)inputs[1]->dataAt<float>(0, 0, 0, 0);
    int activate_output_dim = (int)inputs[2]->dataAt<float>(0, 0, 0, 0);
    if (inputs[0]->count() == 0) {
        outputs[0]->reshape(0, 0, 0, 0);
        return Op::reshape(inputs, outputs);
    }
    int in_dimension = (activate_input_dim == -1) ? in_features_ : activate_input_dim;
    int out_dimension = (activate_output_dim == -1) ? out_features_ : activate_output_dim;
    assert(inputs[0]->head() == 1);
    assert(in_dimension == inputs[0]->dimension());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), out_dimension);
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUElasticLinear::load(AbstructLoader &loader) {
    // std::cout << name() << "  CPUElasticLinear load" << std::endl;
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, out_features_, in_features_);
    if (loader.getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        loader.load(&weight_);
    } else {
        weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
    }
    if (support_bias_) {
        bias_.setName(name() + ".bias");
        bias_.reshape(1, 1, 1, out_features_);
        if (loader.getDataType(bias_.name()) != MLLM_TYPE_COUNT) {
            bias_.setDtype(loader.getDataType(bias_.name()));
            bias_.alloc();
            loader.load(&bias_);
        } else {
            bias_.setDtype(MLLM_TYPE_F32);
            bias_.alloc();
        }
    }
    return Op::load(loader);
}

ErrorCode CPUElasticLinear::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    int activate_input_dim = (int)inputs[1]->dataAt<float>(0, 0, 0, 0);
    int activate_output_dim = (int)inputs[2]->dataAt<float>(0, 0, 0, 0);

    //    auto start = mllm::mllm_time_us();
    if (inputs[0]->count() == 0) {
        return Op::execute(inputs, outputs);
    }
    mat_mul_elastic(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_, activate_input_dim, activate_output_dim, false, true, thread_count);
    return Op::execute(inputs, outputs);
}
ErrorCode CPUElasticLinear::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    if (support_bias_) {
        bias_.free();
    }
    return Op::free(inputs, outputs);
}

} // namespace mllm
