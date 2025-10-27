//
// Created by Rongjie Yi on 24-12-26.
//

#ifndef CPUSCATTE_HPP
#define CPUSCATTE_HPP

#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"
#include <cassert>
#include <iostream>
#include <memory>

namespace mllm {
class Tensor;

class CPUScatter : public Op {
private:
    int thread_count = 4;
    Chl dim_;            // default dimension is SEQUENCE
    float value_ = 0.0f; // default value is 0.0f

public:
    CPUScatter(Backend *bn, string name, Chl dim, float value, int threadCount) :
        Op(bn, name), dim_(dim), value_(value), thread_count(threadCount) {
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        if (inputs[1]->batch() == 0) {
            return MLLM_NO_ERROR;
        }
        assert(inputs.size() == 2);
        assert(inputs[0]->batch() == 1);
        auto dest_input = inputs[0];
        auto replace_idx = inputs[1];
        if (dim_ == SEQUENCE) {
            assert(inputs[0]->head() == 1);
            assert(replace_idx->batch() == 1);
            assert(replace_idx->sequence() == 1);
            assert(replace_idx->head() == 1);
            assert(dest_input->head() == 1);
            // Todo check
            //  #pragma omp parallel for num_threads(CPUBackend::cpu_threads)
            for (int r_idx = 0; r_idx < replace_idx->dimension(); r_idx++) {
                auto replace_seq = (int)replace_idx->dataAt<float>(0, 0, 0, r_idx);
                auto dst_ptr = dest_input->ptrAt<float>(0, 0, replace_seq, 0);
                memset(dst_ptr, value_, sizeof(float) * dest_input->dimension());
            }
        } else if (dim_ == HEAD) {
            assert(replace_idx->sequence() == dest_input->sequence());
            for (int tok = 0; tok < replace_idx->sequence(); tok++) {
                for (int r_idx = 0; r_idx < replace_idx->dimension(); r_idx++) {
                    auto replace_seq = (int)replace_idx->dataAt<float>(0, 0, tok, r_idx);
                    auto dst_ptr = dest_input->ptrAt<float>(0, replace_seq, tok, 0);
                    dest_input->setDataAt<float>(0, replace_seq, tok, 0, value_);
                }
            };
        } else {
            std::cerr << "Error: CPUScatter only supports SEQUENCE dimension currently." << std::endl;
            return NOT_SUPPORT;
        }
        return MLLM_NO_ERROR;
    }
};

class CPUScatterCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        Chl dim = SEQUENCE;
        auto it = op_param.find("dim");
        if (it != op_param.end()) {
            dim = static_cast<Chl>(it->second);
        }
        float value = static_cast<float>(op_param["value"]);
        return new CPUScatter(bn, name, dim, value, threadCount);
    }
};

} // namespace mllm
#endif // CPUSCATTE_HPP