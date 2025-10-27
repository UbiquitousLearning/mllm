//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUINDEXPUTFUNC_HPP
#define CPUINDEXPUTFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"
#include <vector>
#include <memory>

namespace mllm {
class Tensor;

class CPUIndexPutFunction : public Op {
private:
    int thread_count = 4;
    bool accumulate_;

public:
    CPUIndexPutFunction(Backend *bn, string name, int threadCount, bool accumulate) :
        Op(bn, name), thread_count(threadCount), accumulate_(accumulate) {
    }

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        if (!accumulate_) {
            if (inputs[0]->masterTensor() == nullptr) {
                inputs[0]->free();
            }
            outputs[0]->alloc();
            inputs[0]->shallowCopyFrom(outputs[0], false);
        }
        return MLLM_NO_ERROR;
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        if (inputs.size() > 1 && inputs[1]->batch() == 0) {
            outputs[0]->reshape(inputs[0]->batch(), 1, inputs[0]->sequence(), inputs[0]->dimension());
            if (!accumulate_) {
                if (inputs[0]->masterTensor() == nullptr) {
                    inputs[0]->free();
                }
                outputs[0]->alloc();
                inputs[0]->shallowCopyFrom(outputs[0], false);
            }
            return MLLM_NO_ERROR;
        }
        // reshape
        assert(inputs.size() == 3);
        auto dest_input = inputs[0];
        auto src_input = inputs[1];
        assert(dest_input->batch() == 1);
        assert(dest_input->head() == 1);
        assert(src_input->head() == 1);
        assert(dest_input->dimension() == src_input->dimension());

        if (!accumulate_) {
            outputs[0]->reshape(dest_input->batch(), dest_input->head(), dest_input->sequence(), dest_input->dimension());
        } else {
            int origin_s = dest_input->sequence();
            int replace_s = src_input->sequence();
            int replace_size = src_input->batch();
            int seq = origin_s - replace_size + (replace_size * replace_s);
            outputs[0]->reshape(dest_input->batch(), dest_input->head(), seq, dest_input->dimension());
            outputs[0]->alloc();
        }

        outputs[0]->setDtype(inputs[0]->dtype());
        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        if (inputs.size() > 1 && inputs[1]->batch() == 0) {
            return MLLM_NO_ERROR;
        }

        assert(inputs.size() == 3);
        auto dest_input = inputs[0];
        auto src_input = inputs[1];
        auto replace_idx = inputs[2];
        assert(replace_idx->batch() == 1);
        assert(replace_idx->sequence() == 1);
        assert(replace_idx->head() == 1);
        if (!accumulate_) {
            for (int r_idx = 0; r_idx < replace_idx->dimension(); r_idx++) {
                auto replace_seq = (int)replace_idx->dataAt<float>(0, 0, 0, r_idx);
                auto dst_ptr = inputs[0]->ptrAt<float>(0, 0, replace_seq, 0);
                auto src_ptr = src_input->ptrAt<float>(0, 0, r_idx, 0);
                memcpy(dst_ptr, src_ptr, sizeof(float) * src_input->dimension());
            }
        } else if (replace_idx->dimension() == src_input->batch()) {
            int replace_s = src_input->sequence();
            int replace_size = src_input->batch();
            auto start_dest_seq = 0;
            int in0_d = 0;
            int in1_batch = 0;
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
            for (int i = 0; i < replace_size; ++i) {
                auto start_src_seq = (int)replace_idx->dataAt<float>(0, 0, 0, i) + (i * replace_s);
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
        } else {
            for (int r_idx = 0; r_idx < replace_idx->dimension(); r_idx++) {
                auto replace_seq = (int)replace_idx->dataAt<float>(0, 0, 0, r_idx);
                auto dst_ptr = outputs[0]->ptrAt<float>(0, 0, replace_seq, 0);
                auto src_ptr = src_input->ptrAt<float>(0, 0, r_idx, 0);
                memcpy(dst_ptr, src_ptr, sizeof(float) * src_input->dimension());
            }
        }
        return MLLM_NO_ERROR;
    }
};

class CPUIndexPutFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        bool accumulate = (bool)op_param.at("accumulate");
        return new CPUIndexPutFunction(bn, name, threadCount, accumulate);
    }
};

} // namespace mllm
#endif // CPUINDEXPUTFUNC_HPP