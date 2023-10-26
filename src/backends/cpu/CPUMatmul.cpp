
#include "CPUMatmul.hpp"

namespace mllm {

CPUMatmul::CPUMatmul(Backend *bn, string opName, bool transpose0, bool transpose1, bool multiThread) :
    Op(bn, opName) {
    transpose0_ = transpose0;
    transpose1_ = transpose1;
    support_multi_thread_ = multiThread;
}

ErrorCode CPUMatmul::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout<<name() << "  CPUMatmul  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    CHECK_EQ(inputs[0]->head(), inputs[1]->head());
    //    CHECK_EQ(inputs[0]->head(), 1);
    CHECK_EQ(inputs[0]->batch(), inputs[1]->batch());
    if (!transpose0_ && !transpose1_) {
        /*
         N     |    C       |   H                   |  W
         -----------------------------------------------
         batch |out_channel | in_channel            |  1
         -----------------------------------------------
         batch |in_channel  | seq_len               |  1
         -----------------------------------------------
         batch |out_channel | seq_len               |  1
         */
        CHECK_EQ(inputs[0]->dimension(), inputs[1]->sequence());
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[1]->dimension());
    } else if (transpose1_) {
        /*
         N     |    C       |   H                   |  W
         -----------------------------------------------
         batch |in_channel | out_channel            |  1
         -----------------------------------------------
         batch |in_channel  | seq_len               |  1
         -----------------------------------------------
         batch |out_channel | seq_len               |  1
         */
        CHECK_EQ(inputs[0]->dimension(), inputs[1]->dimension());
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[1]->sequence());
    } else {
        /*
         N     |    C       |   H                   |  W
         -----------------------------------------------
         batch |out_channel | in_channel            |  1
         -----------------------------------------------
         batch |seq_len     | in_channel            |  1
         -----------------------------------------------
         batch |out_channel | seq_len               |  1
         */
        CHECK_EQ(inputs[0]->sequence(), inputs[1]->sequence());
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->dimension(), inputs[1]->dimension());
    }
    outputs[0]->setDtype(activationDtype());
    return NO_ERROR;
}

ErrorCode CPUMatmul::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout<<name() << "  CPUMatmul  setUp" << std::endl;
    if (!inputs[0]->allocted()) {
        inputs[0]->alloc(); // TODO remove
    }
    if (!inputs[1]->allocted()) {
        inputs[1]->alloc(); // TODO remove
    }
    outputs[0]->alloc();
    return NO_ERROR;
}

ErrorCode CPUMatmul::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout<<name() << "  CPUMatmul()" << std::endl;

    switch (weightsDtype()) {
    case MLLM_TYPE_F32: {
        mat_mul_fp32(inputs[0].get(), inputs[1].get(), outputs[0].get(), false, nullptr, transpose0_, transpose1_);
        break;
    }
    case MLLM_TYPE_F16: break;
    case MLLM_TYPE_Q4_0: {
        mat_mul_fp32_q4_0(inputs[0].get(), inputs[1].get(), outputs[0].get(), false, nullptr, transpose0_, transpose1_);
        break;
    }
    default:
        break;
    }
    return NO_ERROR;
}

ErrorCode CPUMatmul::load(ParamLoader &loader) {
    std::cout<<name() << "  CPUMatmul load" << std::endl;
    return NO_ERROR;
}

} // namespace mllm
