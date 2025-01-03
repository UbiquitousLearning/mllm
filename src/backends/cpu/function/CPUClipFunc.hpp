//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUCLIPFUNC_HPP
#define CPUCLIPFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"

namespace mllm {
class Tensor;

class CPUclipFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        int b_size = args[0];
        int h_size = args[1];
        int s_size = args[2];
        int d_size = args[3];
        vector<int> b = {};
        vector<int> h = {};
        vector<int> s = {};
        vector<int> d = {};
        for (int i = 0; i < b_size; i++) {
            b.push_back(args[4 + i]);
        }
        for (int i = 0; i < h_size; i++) {
            h.push_back(args[4 + b_size + i]);
        }
        for (int i = 0; i < s_size; i++) {
            s.push_back(args[4 + b_size + h_size + i]);
        }
        for (int i = 0; i < d_size; i++) {
            d.push_back(args[4 + b_size + h_size + s_size + i]);
        }
        int dim_b = inputs[0]->batch();
        int dim_h = inputs[0]->head();
        int dim_s = inputs[0]->sequence();
        int dim_d = inputs[0]->dimension();
        std::vector<std::pair<std::vector<int>, int *>> data = {{b, &dim_b}, {h, &dim_h}, {s, &dim_s}, {d, &dim_d}};
        for (auto &pair : data) {
            if (pair.first.size() == 2) {
                *pair.second = pair.first[1] - pair.first[0];
            } else if (pair.first.size() == 1) {
                *pair.second = 1;
            }
        }
        outputs[0]->reshape(dim_b, dim_h, dim_s, dim_d);
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        int b_size = args[0];
        int h_size = args[1];
        int s_size = args[2];
        int d_size = args[3];
        vector<int> b = {};
        vector<int> h = {};
        vector<int> s = {};
        vector<int> d = {};
        for (int i = 0; i < b_size; i++) {
            b.push_back(args[4 + i]);
        }
        for (int i = 0; i < h_size; i++) {
            h.push_back(args[4 + b_size + i]);
        }
        for (int i = 0; i < s_size; i++) {
            s.push_back(args[4 + b_size + h_size + i]);
        }
        for (int i = 0; i < d_size; i++) {
            d.push_back(args[4 + b_size + h_size + s_size + i]);
        }
        int dim_b = inputs[0]->batch();
        int dim_h = inputs[0]->head();
        int dim_s = inputs[0]->sequence();
        int dim_d = inputs[0]->dimension();
        std::vector<std::pair<std::vector<int>, int *>> data = {{b, &dim_b}, {h, &dim_h}, {s, &dim_s}, {d, &dim_d}};
        for (auto &pair : data) {
            if (pair.first.size() == 2) {
                *pair.second = pair.first[1] - pair.first[0];
            } else if (pair.first.size() == 1) {
                *pair.second = 1;
            }
        }
        if (outputs[0]->dimension() * outputs[0]->sequence() * outputs[0]->head() * outputs[0]->batch() == 0
            || outputs[0]->shape().empty()
            || dim_d != outputs[0]->dimension()) {
            outputs[0]->reshape(dim_b, dim_h, dim_s, dim_d);
            outputs[0]->alloc();
        }

        if (s.size() == 2) {
#pragma omp parallel for collapse(1) num_threads(CPUBackend::cpu_threads)
            for (int b = 0; b < inputs[0]->batch(); ++b) {
                memcpy(outputs[0]->hostPtr<float>() + outputs[0]->offset(b, 0, 0, 0),
                       inputs[0]->hostPtr<float>() + inputs[0]->offset(b, 0, s[0], 0),
                       inputs[0]->head() * (s[1] - s[0]) * inputs[0]->dimension() * sizeof(float));
            }
        } else if (s.size() == 1) {
            int seq_idx = s[0];
            if (seq_idx < 0) {
                seq_idx = inputs[0]->sequence() + seq_idx;
            }
#pragma omp parallel for collapse(1) num_threads(CPUBackend::cpu_threads)
            for (int b = 0; b < inputs[0]->batch(); ++b) {
                memcpy(outputs[0]->hostPtr<float>() + outputs[0]->offset(b, 0, 0, 0),
                       inputs[0]->hostPtr<float>() + inputs[0]->offset(b, 0, seq_idx, 0),
                       inputs[0]->head() * 1 * inputs[0]->dimension() * sizeof(float));
            }
        } else if (b.size() == 1) {
            int bth_idx = b[0];
            if (bth_idx < 0) {
                bth_idx = inputs[0]->batch() + bth_idx;
            }
            memcpy(outputs[0]->hostPtr<float>(),
                   inputs[0]->hostPtr<float>() + inputs[0]->offset(bth_idx, 0, 0, 0),
                   inputs[0]->head() * inputs[0]->sequence() * inputs[0]->dimension() * sizeof(float));
        } else if (b.size() == 2) {
            assert(b[1] - b[0] > 0);
            memcpy(outputs[0]->hostPtr<float>(),
                   inputs[0]->hostPtr<float>() + inputs[0]->offset(b[0], 0, 0, 0),
                   (b[1] - b[0]) * inputs[0]->head() * inputs[0]->sequence() * inputs[0]->dimension() * sizeof(float));
        } else if (d.size() == 2) {
#pragma omp parallel for collapse(1) num_threads(CPUBackend::cpu_threads)
            for (int b = 0; b < inputs[0]->batch(); ++b) {
                for (int s = 0; s < inputs[0]->sequence(); ++s) {
                    for (int h = 0; h < inputs[0]->head(); ++h) {
                        memcpy(outputs[0]->hostPtr<float>() + outputs[0]->offset(b, h, s, 0),
                               inputs[0]->hostPtr<float>() + inputs[0]->offset(b, h, s, d[0]),
                               (d[1] - d[0]) * sizeof(float));
                    }
                }
            }
        } else {
            std::cout << "[TODO]Tensor.CLip not support!!!!" << std::endl;
        }
    }
};

class CPUclipaxisFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        Chl axis = (Chl)args[0];
        int b_size = args[1];
        int h_size = args[2];
        int s_size = args[3];
        int d_size = args[4];
        vector<int> b = {};
        vector<int> h = {};
        vector<int> s = {};
        vector<int> d = {};
        for (int i = 0; i < b_size; i++) {
            b.push_back(args[5 + i]);
        }
        for (int i = 0; i < h_size; i++) {
            h.push_back(args[5 + b_size + i]);
        }
        for (int i = 0; i < s_size; i++) {
            s.push_back(args[5 + b_size + h_size + i]);
        }
        for (int i = 0; i < d_size; i++) {
            d.push_back(args[5 + b_size + h_size + s_size + i]);
        }
        int dim_b = inputs[0]->batch();
        int dim_h = inputs[0]->head();
        int dim_s = inputs[0]->sequence();
        int dim_d = inputs[0]->dimension();
        switch (axis) {
        case BATCH: {
            std::vector<std::pair<std::vector<int>, int *>> data = {{h, &dim_h}, {s, &dim_s}, {d, &dim_d}};
            for (auto &pair : data) {
                if (!pair.first.empty()) {
                    *pair.second = 1;
                }
            }
            break;
        }
        case HEAD: {
            std::vector<std::pair<std::vector<int>, int *>> data = {{b, &dim_b}, {s, &dim_s}, {d, &dim_d}};
            for (auto &pair : data) {
                if (!pair.first.empty()) {
                    *pair.second = 1;
                }
            }
            break;
        }
        case SEQUENCE: {
            std::vector<std::pair<std::vector<int>, int *>> data = {{b, &dim_b}, {h, &dim_h}, {d, &dim_d}};
            for (auto &pair : data) {
                if (!pair.first.empty()) {
                    *pair.second = 1;
                }
            }
            break;
        }
        case DIMENSION: {
            std::vector<std::pair<std::vector<int>, int *>> data = {{b, &dim_b}, {h, &dim_h}, {s, &dim_s}};
            for (auto &pair : data) {
                if (!pair.first.empty()) {
                    *pair.second = 1;
                }
            }
            break;
        }
        default:
            break;
        }
        outputs[0]->reshape(dim_b, dim_h, dim_s, dim_d);
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        Chl axis = (Chl)args[0];
        int b_size = args[1];
        int h_size = args[2];
        int s_size = args[3];
        int d_size = args[4];
        vector<int> b = {};
        vector<int> h = {};
        vector<int> s = {};
        vector<int> d = {};
        for (int i = 0; i < b_size; i++) {
            b.push_back(args[5 + i]);
        }
        for (int i = 0; i < h_size; i++) {
            h.push_back(args[5 + b_size + i]);
        }
        for (int i = 0; i < s_size; i++) {
            s.push_back(args[5 + b_size + h_size + i]);
        }
        for (int i = 0; i < d_size; i++) {
            d.push_back(args[5 + b_size + h_size + s_size + i]);
        }
        if (axis == BATCH) {
            if (!s.empty()) {
                for (int i = 0; i < s.size(); ++i) {
                    auto seq_idx = s[i];
                    memcpy(outputs[0]->hostPtr<float>() + outputs[0]->offset(i, 0, 0, 0),
                           inputs[0]->hostPtr<float>() + inputs[0]->offset(i, 0, seq_idx, 0),
                           inputs[0]->head() * 1 * inputs[0]->dimension() * sizeof(float));
                }
            }
        } else {
            std::cout << "[TODO]Tensor.CLip not support!!!!" << std::endl;
        }
    }
};

class CPUcliptensorFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        Chl dim = (Chl)args[0];
        if (dim == SEQUENCE) {
            int new_seq = inputs[1]->dimension();
            outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), new_seq, inputs[0]->dimension());
        } else {
            std::cout << "[TODO]Tensor.CLip not support!!!!" << std::endl;
        }
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        Chl dim = (Chl)args[0];
        if (dim == SEQUENCE) {
            int new_seq = inputs[1]->dimension();
            if (outputs[0]->sequence() == 0 || outputs[0]->shape().empty()
                || new_seq != outputs[0]->sequence()) {
                outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), new_seq, inputs[0]->dimension());
                outputs[0]->alloc();
            }
            for (int d = 0; d < inputs[1]->dimension(); ++d) {
                auto dim_idx = inputs[1]->dataAt<float>(0, 0, 0, d);
                memcpy(outputs[0]->ptrAt<float>(0, 0, d, 0),
                       inputs[0]->ptrAt<float>(0, 0, (int)dim_idx, 0),
                       inputs[0]->head() * inputs[0]->dimension() * sizeof(float));
            }
        } else {
            std::cout << "[TODO]Tensor.CLip not support!!!!" << std::endl;
        }
    }
};
} // namespace mllm
#endif // CPUCLIPFUNC_HPP