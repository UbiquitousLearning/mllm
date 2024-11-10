#include "backends/xnnpack/Ops/XpEmbedding.hpp"
#include "backends/cpu/quantize/QuantizeQ4.hpp"
#include "backends/cpu/quantize/QuantizeQ8.hpp"

namespace mllm::xnnpack {

ErrorCode XpEmbedding::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::setUp(inputs, outputs);
}

ErrorCode XpEmbedding::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];
    output->reshape(input->batch(), 1, input->sequence(), hidden_size_);
    return Op::reshape(inputs, outputs);
}

ErrorCode XpEmbedding::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    auto &input = inputs[0];
    auto &output = outputs[0];
    switch (weight_params_.dtype()) {
    case MLLM_TYPE_F32: {
        for (int batch = 0; batch < input->batch(); ++batch) {
            for (int head = 0; head < input->head(); ++head) { // NOLINT(*-use-default-none)
#pragma omp parallel for num_threads(thread_count)
                for (int seq = 0; seq < input->sequence(); ++seq) {
#ifdef USE_QNN
                    if ((int)input->dataAt<float>(batch, head, seq, 0) == vocab_size_) {
                        memset(output->hostPtr<float>() + output->offset(batch, head, seq, 0), 0, output->dimension() * sizeof(float));
                        continue;
                    }
#endif
                    memcpy(output->hostPtr<float>() + output->offset(batch, head, seq, 0),
                           weight_params_.hostPtr<float>() + weight_params_.offset(0, 0, (int)input->dataAt<float>(batch, head, seq, 0), 0),
                           weight_params_.dtypeSize() * hidden_size_);
                }
            }
        }
        break;
    }
    case MLLM_TYPE_Q4_0: {
        for (int batch = 0; batch < input->batch(); ++batch) {
            for (int head = 0; head < input->head(); ++head) {
#pragma omp parallel for num_threads(thread_count)
                for (int seq = 0; seq < input->sequence(); ++seq) {
                    dequantize_row_q4_0(weight_params_.hostPtr<block_q4_0>() + weight_params_.offset(0, 0, (int)input->dataAt<float>(batch, head, seq, 0), 0) / (QK4_0),
                                        output->hostPtr<float>() + output->offset(batch, head, seq, 0),
                                        hidden_size_);
                }
            }
        }
        break;
    }
    case MLLM_TYPE_Q4_K: {
        for (int batch = 0; batch < input->batch(); ++batch) {
            for (int head = 0; head < input->head(); ++head) {
#pragma omp parallel for num_threads(thread_count)
                for (int seq = 0; seq < input->sequence(); ++seq) {
                    dequantize_row_q4_K(weight_params_.hostPtr<block_q4_K>() + weight_params_.offset(0, 0, (int)inputs[0]->dataAt<float>(batch, head, seq, 0), 0) / (QK_K),
                                        outputs[0]->hostPtr<float>() + outputs[0]->offset(batch, head, seq, 0),
                                        hidden_size_);
                }
            }
        }
        break;
    }
    case MLLM_TYPE_Q8_0: {
        for (int batch = 0; batch < input->batch(); ++batch) {
            for (int head = 0; head < input->head(); ++head) {
#pragma omp parallel for num_threads(thread_count)
                for (int seq = 0; seq < input->sequence(); ++seq) {
                    dequantize_row_q8_0(weight_params_.hostPtr<block_q8_0>() + weight_params_.offset(0, 0, (int)input->dataAt<float>(batch, head, seq, 0), 0) / (QK8_0),
                                        output->hostPtr<float>() + output->offset(batch, head, seq, 0),
                                        hidden_size_);
                }
            }
        }
        break;
    }
    case MLLM_TYPE_Q8_K: {
        for (int batch = 0; batch < input->batch(); ++batch) {
            for (int head = 0; head < input->head(); ++head) {
#pragma omp parallel for num_threads(thread_count)
                for (int seq = 0; seq < input->sequence(); ++seq) {
                    dequantize_row_q8_K(weight_params_.hostPtr<block_q8_K>() + weight_params_.offset(0, 0, (int)input->dataAt<float>(batch, head, seq, 0), 0) / (QK_K),
                                        output->hostPtr<float>() + output->offset(batch, head, seq, 0),
                                        hidden_size_);
                }
            }
        }
        break;
    }
    case MLLM_TYPE_F16:
    case MLLM_TYPE_Q4_1:
    case MLLM_TYPE_Q8_1:
    case MLLM_TYPE_Q6_K:
    case MLLM_TYPE_I8:
    case MLLM_TYPE_I16:
    case MLLM_TYPE_I32:
    case MLLM_TYPE_COUNT:
    default: break;
    }

    return MLLM_NO_ERROR;
}

ErrorCode XpEmbedding::load(AbstructLoader &loader) {
    weight_params_.setName(name() + ".weight");
    weight_params_.reshape(1, 1, vocab_size_, hidden_size_);
    if (loader.getDataType(weight_params_.name()) != MLLM_TYPE_COUNT) {
        weight_params_.setDtype(loader.getDataType(weight_params_.name()));
        weight_params_.alloc();
        loader.load(&weight_params_);
    } else {
        weight_params_.setDtype(MLLM_TYPE_F32);
        weight_params_.alloc();
    }
    return Op::load(loader);
}

ErrorCode XpEmbedding::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_params_.free();
    return MLLM_NO_ERROR;
}

Op *XpEmbeddingCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    auto hidden_size = int(op_param["hidden_size"]);
    auto vocab_size = int(op_param["vocab_size"]);
    return new XpEmbedding(bk, name, hidden_size, vocab_size, thread_count);
}
} // namespace mllm::xnnpack
