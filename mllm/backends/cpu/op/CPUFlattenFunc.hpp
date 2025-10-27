//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUFLATTENFUNC_HPP
#define CPUFLATTENFUNC_HPP

#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"
#include <iostream>
#include <vector>
#include "Module.hpp"

namespace mllm {
class Tensor;

class CPUflattenFunction : public Op {
private:
    int thread_count = 4;
    Chl axis_start_;
    Chl axis_end_;

public:
    CPUflattenFunction(Backend *bn, string name, int threadCount, Chl axis_start, Chl axis_end) :
        Op(bn, name), thread_count(threadCount), axis_start_(axis_start), axis_end_(axis_end) {
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        auto input = inputs[0];
        auto output = outputs[0];

        int dim_b = input->batch();
        int dim_h = 0;
        int dim_s = 0;
        int dim_d = 0;

        if (inputs[0]->shape().size() == 4) {
            dim_h = inputs[0]->head();
            dim_s = inputs[0]->sequence();
            dim_d = inputs[0]->dimension();
            if (axis_start_ == BATCH & axis_end_ == SEQUENCE) {
                dim_b = 1;
                dim_s = inputs[0]->sequence() * inputs[0]->batch();
            } else if (axis_start_ == HEAD & axis_end_ == SEQUENCE) {
                dim_h = 1;
                dim_s = inputs[0]->sequence() * inputs[0]->head();
            } else if (axis_start_ == HEAD & axis_end_ == DIMENSION) {
                dim_h = 1;
                dim_d = inputs[0]->dimension() * inputs[0]->head();
            } else {
                std::cout << "ERROR:  flatten  " << axis_start_ << "&" << axis_end_ << std::endl;
            }
        } else if (inputs[0]->shape().size() == 5) {
            if (axis_start_ == CHANNLE & axis_end_ == HEIGHT) {
                dim_h = 1;
                dim_s = inputs[0]->channel() * inputs[0]->height() * inputs[0]->time();
                dim_d = inputs[0]->width();
            } else if (axis_start_ == HEIGHT & axis_end_ == CHANNLE) {
                dim_h = 1;
                dim_s = inputs[0]->channel() * inputs[0]->height() * inputs[0]->width();
                dim_d = inputs[0]->time();
            }
        }
        assert(dim_d + dim_s + dim_h > 0);
        if (inputs[0]->ctype() == BCTHW) { // TODOTMPA
            outputs[0]->chls()[BATCH] = 0;
            outputs[0]->chls()[SEQUENCE] = 1;
            outputs[0]->chls()[HEAD] = 2;
            outputs[0]->chls()[DIMENSION] = 3;
            outputs[0]->setCtype(BSHD);
        }
        outputs[0]->reshape(dim_b, dim_h, dim_s, dim_d);
        return ErrorCode::MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        // No data movement needed, all work done in reshape by creating a view.
        return ErrorCode::MLLM_NO_ERROR;
    }

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        // inputs[0]->shallowCopyFrom(outputs[0].get(), false);
        // Chl axis_start = (Chl)args[0];
        // Chl axis_end = (Chl)args[1];
        if ((axis_start_ == TIME & axis_end_ == WIDTH && inputs[0]->ctype() == BCTHW)
            || (axis_start_ == CHANNLE & axis_end_ == HEIGHT && inputs[0]->ctype() == BWCTH)
            || (axis_start_ == HEIGHT & axis_end_ == CHANNLE && inputs[0]->ctype() == BTHWC)
            || (axis_start_ == BATCH & axis_end_ == SEQUENCE && inputs[0]->ctype() != BCTHW)
            || (axis_start_ == HEAD & axis_end_ == SEQUENCE && inputs[0]->ctype() == BSHD)
            || (axis_start_ == HEAD & axis_end_ == SEQUENCE && inputs[0]->ctype() == BHDS)
            || (axis_start_ == HEAD & axis_end_ == DIMENSION && inputs[0]->ctype() == BSHD)
            || (axis_start_ == HEAD & axis_end_ == DIMENSION && inputs[0]->ctype() == BHDS)
            || (axis_start_ == HEAD & axis_end_ == SEQUENCE && inputs[0]->ctype() == BDSH)) {
            if (inputs[0]->masterTensor() == nullptr) {
                inputs[0]->free();
            }
            outputs[0]->setDtype(inputs[0]->dtype());
            outputs[0]->alloc();
            inputs[0]->shallowCopyFrom(outputs[0], false);
        } else if (Module::llm_model_ptr->op_transposed_flag) {
            if (inputs[0]->masterTensor() == nullptr) {
                inputs[0]->free();
            }
            outputs[0]->setDtype(inputs[0]->dtype());
            outputs[0]->alloc();
            inputs[0]->shallowCopyFrom(outputs[0], false);
        } else {
            std::cout << "[TODO]Tensor.Flatten not support!!!!" << std::endl;
        }
        return ErrorCode::MLLM_NO_ERROR;
    }
};

class CPUflattenFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // Assumes OpParam contains keys "axis_start" and "axis_end"
        Chl axis_start = (Chl)op_param.at("axis_start");
        Chl axis_end = (Chl)op_param.at("axis_end");
        return new CPUflattenFunction(bn, name, threadCount, axis_start, axis_end);
    }
};

} // namespace mllm
#endif // CPUFLATTENFUNC_HPP