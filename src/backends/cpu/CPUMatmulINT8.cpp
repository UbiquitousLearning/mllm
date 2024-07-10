
#include "CPUMatmulINT8.hpp"
#include "Types.hpp"
#include "compute/Matmul.hpp"
#include "compute/StrassenMatmul.hpp"
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>

namespace mllm {

CPUMatmulINT8::CPUMatmulINT8(Backend *bn, string opName, bool transpose0, bool transpose1, int threadCount) :
    thread_count(threadCount),
    matmul_(backend_, 5, threadCount),
    Op(bn, opName) {
    transpose0_ = transpose0;
    transpose1_ = transpose1;
    thread_count = threadCount;
    matmul_vec_.resize(threadCount, std::make_shared<StrassenMatmul>(StrassenMatmul(backend_, 5, threadCount)));
}

ErrorCode CPUMatmulINT8::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);
    assert(inputs[0]->head() == inputs[1]->head());

    assert(inputs[0]->head() % thread_count == 0);
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

    matmul_.onReset();
    for (auto &mat : matmul_vec_) {
        mat->onReset();
    }

    inputs_a_.clear();
    inputs_a_.resize(thread_count);
    inputs_b_.clear();
    inputs_b_.resize(thread_count);
    outputs_.clear();
    outputs_.resize(thread_count);

    return Op::reshape(inputs, outputs);
}

ErrorCode CPUMatmulINT8::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (!isInitialized) {
        isInitialized = true;
        Op::setUp(inputs, outputs);

        float scale1_value = scale1_.dataAt<float>(0, 0, 0, 0) / 127.0;
        scale1_value = roundf(scale1_value * 10000) / 10000;
        float scale2_value = scale2_.dataAt<float>(0, 0, 0, 0) / 127.0;
        scale2_value = roundf(scale2_value * 10000) / 10000;

        std::cout << "input0:" << inputs[0]->dtype() << std::endl;
        std::cout << "input1:" << inputs[1]->dtype() << std::endl;

        int headGroup = inputs[0]->head() / thread_count;

        for (int i = 0; i < inputs_a_.size(); ++i) {
            inputs_a_[i] = std::make_shared<Tensor>(backend_);
            inputs_a_[i]->setDtype(inputs[0]->dtype());
            inputs_a_[i]->setCtype(inputs[0]->ctype());
            inputs_a_[i]->setBackend(backend_);
            inputs_a_[i]->reshape(inputs[0]->batch(), headGroup, inputs[0]->sequence(), inputs[0]->dimension());
            inputs_a_[i]->deepCopyFrom(inputs[0].get(), false, {0, 0, headGroup * i, 0});

            inputs_b_[i] = std::make_shared<Tensor>(backend_);
            inputs_b_[i]->setDtype(inputs[1]->dtype());
            inputs_b_[i]->setCtype(inputs[1]->ctype());
            inputs_b_[i]->setBackend(backend_);
            inputs_b_[i]->reshape(inputs[1]->batch(), headGroup, inputs[1]->sequence(), inputs[1]->dimension());
            inputs_b_[i]->deepCopyFrom(inputs[1].get(), false, {0, 0, headGroup * i, 0});

            outputs_[i] = std::make_shared<Tensor>(backend_);
            outputs_[i]->setDtype(outputs[0]->dtype());
            outputs_[i]->setCtype(outputs[0]->ctype());
            outputs_[i]->setBackend(backend_);
            outputs_[i]->reshape(outputs[0]->batch(), headGroup, outputs[0]->sequence(), outputs[0]->dimension());
            outputs_[i]->deepCopyFrom(outputs[0].get(), false, {0, 0, headGroup * i, 0});

            matmul_vec_[i]->onReshape(inputs_a_[i].get(), inputs_b_[i].get(), outputs_[i].get(), false, nullptr, transpose0_, transpose1_, scale1_value, scale2_value);
        }

        // return matmul_.onReshape(inputs[0].get(), inputs[1].get(), outputs[0].get(), false, nullptr, transpose0_, transpose1_, scale1_value, scale2_value);
        return MLLM_NO_ERROR;
    } else {
        return Op::setUp(inputs, outputs);
    }
}

ErrorCode CPUMatmulINT8::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs[1]->dtype() == MLLM_TYPE_I8);

    // switch (inputs[0]->dtype()) {
    // case MLLM_TYPE_I8: { // q * k
    //     // NSHD
    //     float scale1_value = scale1_.dataAt<float>(0, 0, 0, 0) / 127.0;
    //     scale1_value = roundf(scale1_value * 10000) / 10000;

    //     float scale2_value = scale2_.dataAt<float>(0, 0, 0, 0) / 127.0;
    //     scale2_value = roundf(scale2_value * 10000) / 10000;

    //     std::cout << scale1_value << " " << scale2_value << std::endl;

    //     mat_mul_i8(inputs[0].get(), inputs[1].get(), outputs[0].get(), false, nullptr, transpose0_, transpose1_, thread_count, scale1_value, scale2_value);
    // } break;
    // case MLLM_TYPE_F32: { // qk * v
    //     float scale_value = scale2_.dataAt<float>(0, 0, 0, 0) / 127.0;
    //     scale_value = roundf(scale_value * 10000) / 10000;

    //     mat_mul_fp32_i8(inputs[0].get(), inputs[1].get(), outputs[0].get(), false, nullptr, transpose0_, transpose1_, thread_count, scale_value);
    // } break;
    // default:
    //     break;
    // }

    for(auto& mat : matmul_vec_) {
        mat->onExecute();
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

        scale2_.setName(scaleName + ".k_proj.output_scale");
        scale2_.reshape(1, 1, 1, 1);
        scale2_.setBackend(this->backend());
        scale2_.setDtype(MLLM_TYPE_F32);
        scale2_.alloc();
        loader.load(&scale2_);
    } else { // qkv
        scaleName.erase(pos, wordToRemove.length());
        scale1_.setName(scaleName + ".q_proj.output_scale");
        scale1_.reshape(1, 1, 1, 1);
        scale1_.setBackend(this->backend());
        scale1_.setDtype(MLLM_TYPE_F32);
        scale1_.alloc();

        scale2_.setName(scaleName + ".v_proj.output_scale");
        scale2_.reshape(1, 1, 1, 1);
        scale2_.setBackend(this->backend());
        scale2_.setDtype(MLLM_TYPE_F32);
        scale2_.alloc();
        loader.load(&scale2_);
    }

    return Op::load(loader);
}

ErrorCode CPUMatmulINT8::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    scale1_.free();
    scale2_.free();
    return Op::free(inputs, outputs);
}

} // namespace mllm
