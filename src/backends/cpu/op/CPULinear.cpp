
#include "CPULinear.hpp"
#include "Types.hpp"
#include <cstddef>
#include <iostream>

namespace mllm {

CPULinear::CPULinear(Backend *bn, string opName, int in_features, int out_features, bool bias, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    in_features_ = in_features;
    out_features_ = out_features;
    support_bias_ = bias;
    thread_count = threadCount;
    weight_.setBackend(bn);
    bias_.setBackend(bn);
}

ErrorCode CPULinear::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << name() << "  CPULinear  reshape" << std::endl;
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    if (inputs[0]->count() == 0 && inputs[0]->sequence() != 0) {
        outputs[0]->reshape(0, 0, 0, 0);
        return Op::reshape(inputs, outputs);
    }
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
    assert(inputs[0]->head() == 1);
    assert(in_features_ == inputs[0]->dimension());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), out_features_);
    // outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPULinear::load(AbstructLoader &loader) {
    // std::cout << name() << "  CPULinear load" << std::endl;
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, out_features_, in_features_);
    if (loader.getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        loader.load(&weight_);
    } else {
        if (weight_.name().find('v') != std::string::npos && Op::noLoadWeightsDtype() == MLLM_TYPE_Q4_0_4_4) {
            weight_.setDtype(MLLM_TYPE_Q4_0);
        } else {
            weight_.setDtype(Op::noLoadWeightsDtype());
        }
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

ErrorCode CPULinear::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //    auto start = mllm::mllm_time_us();
    if (inputs[0]->count() == 0) {
        return Op::execute(inputs, outputs);
    }
    if (inputs[0]->sequence() != outputs[0]->sequence() && outputs[0]->masterTensor() == nullptr) {
        outputs[0]->reshape(outputs[0]->batch(), outputs[0]->head(), inputs[0]->sequence(), outputs[0]->dimension());
        // outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), out_features_);
        outputs[0]->alloc();
    }
    // TODO: Q8_0 KVCache can not use!!
    if (outputs[0]->dtype() == MLLM_TYPE_Q8_0) {
        auto tmp_out = std::make_shared<Tensor>(outputs[0]->backend());
        // tmp_out->setBackend(outputs[0]->backend());
        auto b = outputs[0]->batch();
        auto h = outputs[0]->head();
        auto d = outputs[0]->dimension();
        auto s = outputs[0]->sequence();
        tmp_out->chls() = outputs[0]->chls();
        tmp_out->setCtype(outputs[0]->ctype());
        tmp_out->reshape(b, h, s, d);
        tmp_out->setDtype(MLLM_TYPE_F32);
        tmp_out->alloc();
        mat_mul(inputs[0].get(), &weight_, tmp_out.get(), support_bias_, &bias_, false, true, thread_count);
        if (tmp_out->ctype() == BSHD) {
#pragma omp parallel for collapse(3) num_threads(thread_count)
            for (int b = 0; b < tmp_out->batch(); b++) {
                for (int h = 0; h < tmp_out->head(); h++) {
                    for (int s = 0; s < tmp_out->sequence(); s++) {
                        quantize_row_q8_0(tmp_out->hostPtr<float>() + tmp_out->offset(b, h, s, 0),
                                          (char *)outputs[0]->rawHostPtr()
                                              + outputs[0]->offset(b, h, s, 0) * sizeof(block_q8_0) / QK8_0,
                                          tmp_out->dimension());
                    }
                }
            }
        } else { // BHDS
#pragma omp parallel for collapse(3) num_threads(thread_count)
            for (int b = 0; b < tmp_out->batch(); b++) {
                for (int h = 0; h < tmp_out->head(); h++) {
                    for (int d = 0; d < tmp_out->dimension(); d++) {
                        quantize_row_q8_0(tmp_out->hostPtr<float>() + tmp_out->offset(b, h, 0, d),
                                          (char *)outputs[0]->rawHostPtr()
                                              + outputs[0]->offset(b, h, 0, d) * sizeof(block_q8_0) / QK8_0,
                                          outputs[0]->sequence());
                    }
                }
            }
        }
    } else {
        mat_mul(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_, false, true, thread_count);
    }
    // std::cout << name() << "  CPULinear()" << std::endl;
    /*
    switch (weight_.dtype()) {
    case MLLM_TYPE_F32: {
        mat_mul_fp32(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_, false, true, thread_count);
        break;
    }
    case MLLM_TYPE_F16: break;
    case MLLM_TYPE_Q4_0: {
        mat_mul_fp32_q4_0(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_, thread_count);
        break;
    }
    case MLLM_TYPE_Q4_K: {
        mat_mul_fp32_q4_K(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_, thread_count);
        break;
    }
    case MLLM_TYPE_Q6_K: {
        mat_mul_fp32_q6_K(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_, thread_count);
        break;
    }
    default:
        break;
    }
    */
    //    auto end = mllm::mllm_time_us();
    //    printf("exec time: %ld us\n", end - start);
    return Op::execute(inputs, outputs);
}
ErrorCode CPULinear::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    if (support_bias_) {
        bias_.free();
    }
    return Op::free(inputs, outputs);
}

} // namespace mllm
