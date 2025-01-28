
#include "CPUReplace.hpp"

namespace mllm {

CPUReplace::CPUReplace(Backend *bn, string opName, int accumulate, int threadCount) :
    thread_count(threadCount),
    accumulate(accumulate),
    Op(bn, opName) {
}

ErrorCode CPUReplace::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (inputs[1]->batch() == 0) {
        outputs[0]->reshape(inputs[0]->batch(), 1, inputs[0]->sequence(), inputs[0]->dimension());
        return Op::execute(inputs, outputs);
    }
    assert(inputs.size() == 3);
    assert(outputs.size() == 1);
    auto dest_input = inputs[0];
    auto src_input = inputs[1];
    // auto replace_idx = inputs[2];
    assert(dest_input->batch() == 1);
    assert(dest_input->head() == 1);
    assert(src_input->head() == 1);
    assert(dest_input->dimension() == src_input->dimension());
    int origin_s = dest_input->sequence();
    int replace_s = src_input->sequence();
    int replace_size = src_input->batch();
    int seq = origin_s - replace_size + replace_size * replace_s;
    if (!accumulate) {
        outputs[0]->reshape(dest_input->batch(), dest_input->head(), dest_input->sequence(), dest_input->dimension()); // 1, 1, 595, 4096
    } else {
        outputs[0]->reshape(dest_input->batch(), dest_input->head(), seq, dest_input->dimension()); // 1, 1, 595, 4096
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUReplace::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (inputs[1]->batch() == 0) {
        auto dst_ptr = outputs[0]->ptrAt<float>(0, 0, 0, 0);
        auto src_ptr = inputs[0]->ptrAt<float>(0, 0, 0, 0);
        memcpy(dst_ptr, src_ptr, sizeof(float) * inputs[0]->dimension() * inputs[0]->sequence() * inputs[0]->batch() * inputs[0]->head());
        return Op::execute(inputs, outputs);
    }
    assert(inputs.size() == 3);
    assert(outputs.size() == 1);
    auto dest_input = inputs[0];
    auto src_input = inputs[1];
    auto replace_idx = inputs[2];
    assert(replace_idx->batch() == 1);
    assert(replace_idx->sequence() == 1);
    assert(replace_idx->head() == 1);
    if (replace_idx->dimension() == src_input->batch()) {
        int replace_s = src_input->sequence();
        int replace_size = src_input->batch();
        auto start_dest_seq = 0;
        int in0_d = 0;
        int in1_batch = 0;
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
        for (int i = 0; i < replace_size; ++i) {
            auto start_src_seq = (int)replace_idx->dataAt<float>(0, 0, 0, i) + i * replace_s;
            auto end_dest_seq = start_src_seq;
            auto end_src_seq = start_src_seq + replace_s;

            auto dst_ptr = outputs[0]->ptrAt<float>(0, 0, start_dest_seq, 0);
            auto src_ptr = inputs[0]->ptrAt<float>(0, 0, in0_d, 0);
            memcpy(dst_ptr, src_ptr, sizeof(float) * dest_input->dimension() * (end_dest_seq - start_dest_seq));
            in0_d += end_dest_seq - start_dest_seq;

            dst_ptr = outputs[0]->ptrAt<float>(0, 0, start_src_seq, 0);
            src_ptr = inputs[1]->ptrAt<float>(0, 0, 0, 0);
            memcpy(dst_ptr, src_ptr, sizeof(float) * src_input->dimension() * replace_s);
            in1_batch++;
            in0_d += 1;

            start_dest_seq = end_src_seq;
        }
        auto dst_ptr = outputs[0]->ptrAt<float>(0, 0, start_dest_seq, 0);
        auto src_ptr = inputs[0]->ptrAt<float>(0, 0, in0_d, 0);
        memcpy(dst_ptr, src_ptr, sizeof(float) * dest_input->dimension() * (outputs[0]->sequence() - start_dest_seq));
    } else if (replace_idx->dimension() == src_input->sequence()) {
        for (int r_idx = 0; r_idx < replace_idx->dimension(); r_idx++) {
            auto replace_seq = (int)replace_idx->dataAt<float>(0, 0, 0, r_idx);
            auto dst_ptr = outputs[0]->ptrAt<float>(0, 0, replace_seq, 0);
            auto src_ptr = src_input->ptrAt<float>(0, 0, r_idx, 0);
            memcpy(dst_ptr, src_ptr, sizeof(float) * src_input->dimension());
        }
    }

    return Op::execute(inputs, outputs);
}

ErrorCode CPUReplace::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 3);
    assert(outputs.size() == 1);
    auto dest_input = inputs[0];
    auto src_input = inputs[1];
    auto replace_idx = inputs[2];
    if (!accumulate) {
        if (inputs[0]->masterTensor() == nullptr) {
            inputs[0]->free();
        }
        outputs[0]->setDtype(activation_dtype());
        outputs[0]->alloc();
        inputs[0]->shallowCopyFrom(outputs[0].get(), false);

        return MLLM_NO_ERROR;
    } else {
        return Op::setUp(inputs, outputs);
    }
}

} // namespace mllm
