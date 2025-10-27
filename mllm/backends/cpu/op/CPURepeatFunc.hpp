//
// Created by Rongjie Yi on 24-12-16.
//

#ifndef CPUREPEATEFUNC_HPP
#define CPUREPEATEFUNC_HPP

#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"
#include <cassert>
#include <iostream>
// #include <vector>
#include <memory>

namespace mllm {
class Tensor;

class CPUrepeatFunction : public Op {
private:
    int thread_count = 4;
    Chl dim_;
    int size_;

public:
    CPUrepeatFunction(Backend *bn, string name, int threadCount, Chl dim, int size)
        : Op(bn, name), thread_count(threadCount), dim_(dim), size_(size) {}

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        int batch = inputs[0]->batch();
        int head = inputs[0]->head();
        int sequence = inputs[0]->sequence();
        int dimension = inputs[0]->dimension();

        switch (dim_) {
        case Chl::BATCH:
            batch = size_;
            break;
        case Chl::HEAD:
            head = size_;
            break;
        case Chl::SEQUENCE:
            sequence = size_;
            break;
        case Chl::DIMENSION:
            dimension = size_;
            break;
        default:
            break;
        }

        outputs[0]->reshape(batch, head, sequence, dimension);
        outputs[0]->setDtype(inputs[0]->dtype());
        // 遵从原始 reshape 逻辑，在这里 alloc
        // outputs[0]->alloc();
        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        switch (dim_) {
        case Chl::BATCH: {
            std::cerr << "Repeat Not implemented for BATCH" << std::endl;
            break;
        }
        case Chl::HEAD: {
            std::cerr << "Repeat Not implemented for HEAD" << std::endl;
            break;
        }
        case Chl::SEQUENCE: {
            std::cerr << "Repeat Not implemented for SEQUENCE" << std::endl;
            break;
        }
        case Chl::DIMENSION: {
#pragma omp parallel for collapse(3) num_threads(thread_count)
            for (int b = 0; b < inputs[0]->batch(); b++) {
                for (int h = 0; h < inputs[0]->head(); h++) {
                    for (int s = 0; s < inputs[0]->sequence(); s++) {
                        // Assuming the input dimension to repeat is 1
                        float data = inputs[0]->dataAt<float>(b, h, s, 0);
                        for (int d = 0; d < size_; d++) {
                            outputs[0]->setDataAt<float>(b, h, s, d, data);
                        }
                    }
                }
            }
            break;
        }
        default:
            break;
        }
        return MLLM_NO_ERROR;
    }
};

class CPUrepeatFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // Assumes OpParam contains keys "dim" and "size"
        Chl dim = (Chl)op_param.at("dim");
        int size = static_cast<int>(op_param.at("size"));
        return new CPUrepeatFunction(bn, name, threadCount, dim, size);
    }
};

} // namespace mllm
#endif // CPUREPEATEFUNC_HPP