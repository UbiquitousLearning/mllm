#include "CPUEmbedding.hpp"
#include "ParamLoader.hpp"
#include "quantize/QuantizeQ4.hpp"
#include "quantize/QuantizeQ8.hpp"

namespace mllm {
CPUEmbedding::CPUEmbedding(Backend *bn,  string opName, int hiddenSize, int vocabSize, int threadCount) : thread_count(threadCount),
    Op(bn, opName), hiddenSize_(hiddenSize), vocabSize_(vocabSize) {
    assert(hiddenSize_ > 0);
    assert(vocabSize_ > 0);
    weight_.setBackend(bn);
}
ErrorCode CPUEmbedding::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    auto input = inputs[0];
    auto output = outputs[0];
    // Input: [batch, 1, sequence, 1]
    output->reshape(input->batch(), 1, input->sequence(), hiddenSize_);
    //outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUEmbedding::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, vocabSize_, hiddenSize_);
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
ErrorCode CPUEmbedding::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    auto &input = inputs[0];
    auto &output = outputs[0];
    switch (weight_.dtype()) {
    case MLLM_TYPE_F32: {
        for (int batch = 0; batch < input->batch(); ++batch) {
            for (int head = 0; head < input->head(); ++head) { // NOLINT(*-use-default-none)
                #pragma omp parallel for num_threads(thread_count)
                for (int seq = 0; seq < input->sequence(); ++seq) {
                    memcpy(output->hostPtr<float>() + output->offset(batch, head, seq, 0),
                           weight_.hostPtr<float>() + weight_.offset(0, 0, (int)input->dataAt<float>(batch, head, seq, 0), 0),
                           weight_.dtypeSize() * hiddenSize_);
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
                    dequantize_row_q4_0(weight_.hostPtr<block_q4_0>() + weight_.offset(0, 0, (int)input->dataAt<float>(batch, head, seq, 0), 0)/(QK4_0),
                                        output->hostPtr<float>() + output->offset(batch, head, seq, 0),
                                        hiddenSize_);
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
                    dequantize_row_q4_K(weight_.hostPtr<block_q4_K>() + weight_.offset(0, 0, (int)inputs[0]->dataAt<float>(batch, head, seq, 0), 0)/(QK_K),
                                        outputs[0]->hostPtr<float>() + outputs[0]->offset(batch, head, seq, 0),
                                        hiddenSize_);
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
                    dequantize_row_q8_0(weight_.hostPtr<block_q8_0>() + weight_.offset(0, 0, (int)input->dataAt<float>(batch, head, seq, 0), 0)/(QK8_0),
                                        output->hostPtr<float>() + output->offset(batch, head, seq, 0),
                                        hiddenSize_);
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
                    dequantize_row_q8_K(weight_.hostPtr<block_q8_K>() + weight_.offset(0, 0, (int)input->dataAt<float>(batch, head, seq, 0), 0)/(QK_K),
                                        output->hostPtr<float>() + output->offset(batch, head, seq, 0),
                                        hiddenSize_);
                }
            }
        }
        break;
    }
    case MLLM_TYPE_F16: break;
    case MLLM_TYPE_Q4_1: break;
    case MLLM_TYPE_Q8_1: break;
    case MLLM_TYPE_Q6_K: break;
    case MLLM_TYPE_I8: break;
    case MLLM_TYPE_I16: break;
    case MLLM_TYPE_I32: break;
    case MLLM_TYPE_COUNT: break;
    default: break;
    }
    return MLLM_NO_ERROR;
}
ErrorCode CPUEmbedding::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    return Op::free(inputs, outputs);
}
} // namespace mllm