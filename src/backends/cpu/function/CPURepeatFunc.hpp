//
// Created by Rongjie Yi on 24-12-16.
//

#ifndef CPUREPEATEFUNC_HPP
#define CPUREPEATEFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"
#include <cassert>
#include <iostream>
#include "CPUBackend.hpp"

namespace mllm {
class Tensor;

class CPUrepeatFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        assert(args.size() == 2);
        Chl dim = (Chl)args[0];
        int size = (int)args[1];
        int batch = inputs[0]->batch();
        int head = inputs[0]->head();
        int sequence = inputs[0]->sequence();
        int dimension = inputs[0]->dimension();
        switch (dim) {
        case Chl::BATCH: {
            batch = size;
            break;
        }
        case Chl::HEAD: {
            head = size;
            break;
        }
        case Chl::SEQUENCE: {
            sequence = size;
            break;
        }
        case Chl::DIMENSION: {
            dimension = size;
            break;
        }
        default:
            break;
        }
        outputs[0]->reshape(batch, head, sequence, dimension);
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        assert(args.size() == 2);
        Chl dim = (Chl)args[0];
        int size = (int)args[1];
        switch (dim) {
        case Chl::BATCH: {
            std::cerr << "Repeat Not implemented" << std::endl;
            break;
        }
        case Chl::HEAD: {
            std::cerr << "Repeat Not implemented" << std::endl;
            break;
        }
        case Chl::SEQUENCE: {
            std::cerr << "Repeat Not implemented" << std::endl;
            break;
        }
        case Chl::DIMENSION: {
#pragma omp parallel for collapse(4) num_threads(CPUBackend::cpu_threads)
            for (int b = 0; b < inputs[0]->batch(); b++) {
                for (int s = 0; s < inputs[0]->sequence(); s++) {
                    for (int h = 0; h < inputs[0]->head(); h++) {
                        for (int d = 0; d < size; d++) {
                            float data = inputs[0]->dataAt<float>(b, h, s, 0);
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
    }
};
} // namespace mllm
#endif // CPUREPEATEFUNC_HPP