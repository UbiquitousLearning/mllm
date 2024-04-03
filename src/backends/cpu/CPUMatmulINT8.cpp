
#include "CPUMatmulINT8.hpp"
#include "Types.hpp"
#include "compute/Matmul.hpp"
#include <cstdint>
#include <iostream>

namespace mllm {

CPUMatmulINT8::CPUMatmulINT8(Backend *bn, string opName, bool transpose0, bool transpose1, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    transpose0_ = transpose0;
    transpose1_ = transpose1;
    thread_count = threadCount;
}

ErrorCode CPUMatmulINT8::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);
    assert(inputs[0]->head() == inputs[1]->head());
    //    assert(inputs[0]->head() == 1);
    // assert(inputs[0]->batch() == inputs[1]->batch());
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
        assert(inputs[0]->dimension() == inputs[1]->sequence());
        inputs[1]->transShape(SEQUENCE, DIMENSION);
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
        assert(inputs[0]->dimension() == inputs[1]->dimension());
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
        assert(inputs[0]->sequence() == inputs[1]->sequence());
        inputs[0]->transShape(SEQUENCE, DIMENSION);
        inputs[1]->transShape(SEQUENCE, DIMENSION);
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->dimension(), inputs[1]->dimension());
    }
    // outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUMatmulINT8::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs[1]->dtype() == MLLM_TYPE_I8);

    switch (inputs[0]->dtype()) {
    case MLLM_TYPE_I8: // q * k
        mat_mul_i8(inputs[0].get(), inputs[1].get(), outputs[0].get(), false, nullptr, transpose0_, transpose1_, thread_count, scale1_.dataAt<float>(0, 0, 0, 0), scale2_.dataAt<float>(0, 0, 0, 0));
        break;
    case MLLM_TYPE_F32: // qk * v
        mat_mul_fp32_i8(inputs[0].get(), inputs[1].get(), outputs[0].get(), false, nullptr, transpose0_, transpose1_, thread_count, scale2_.dataAt<float>(0, 0, 0, 0));
        break;
    default:
        break;
    }

    return Op::execute(inputs, outputs);
}

ErrorCode CPUMatmulINT8::load(AbstructLoader &loader) {
    std::cout << name() << "  CPUMatmulInt8 load" << std::endl;
    std::string scaleName = name();
    std::string wordToRemove = ".qkv";
    int pos = scaleName.find(wordToRemove);
    if (pos == -1) { // qk
        wordToRemove = ".qk";
        pos = scaleName.find(wordToRemove);
        scaleName.erase(pos, wordToRemove.length());

        scale1_.setName(scaleName + ".q_proj.output_scale");
        scale1_.reshape(1, 1, 1, 1);
        scale1_.setBackend(this->backend());
        scale1_.setDtype(MLLM_TYPE_F32);
        scale1_.alloc();
        loader.load(&scale1_);
        scale1_.printData<float>();

        scale2_.setName(scaleName + ".k_proj.output_scale");
        scale2_.reshape(1, 1, 1, 1);
        scale2_.setBackend(this->backend());
        scale2_.setDtype(MLLM_TYPE_F32);
        scale2_.alloc();
        loader.load(&scale2_);
        scale2_.printData<float>();
    } else { // qkv
        scaleName.erase(pos, wordToRemove.length());

        scale2_.setName(scaleName + ".v_proj.output_scale");
        scale2_.reshape(1, 1, 1, 1);
        scale2_.setBackend(this->backend());
        scale2_.setDtype(MLLM_TYPE_F32);
        scale2_.alloc();
        loader.load(&scale2_);
        scale2_.printData<float>();
    }

    return Op::load(loader);
}

ErrorCode CPUMatmulINT8::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    scale1_.free();
    scale2_.free();
    return Op::free(inputs, outputs);
}

} // namespace mllm
