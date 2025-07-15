//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUCLIPFUNC_HPP
#define CPUCLIPFUNC_HPP

#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"
#include <vector>
#include <iostream>
#include <utility>

namespace mllm {
class Tensor;

class CPUclipFunction : public Op {
private:
    int thread_count = 4;
    std::vector<int> b_;
    std::vector<int> h_;
    std::vector<int> s_;
    std::vector<int> d_;

public:
    CPUclipFunction(Backend *bn, string name, int threadCount,
                    const std::vector<int> &b, const std::vector<int> &h,
                    const std::vector<int> &s, const std::vector<int> &d) :
        Op(bn, name),
        thread_count(threadCount), b_(b), h_(h), s_(s), d_(d) {
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        int dim_b = inputs[0]->batch();
        int dim_h = inputs[0]->head();
        int dim_s = inputs[0]->sequence();
        int dim_d = inputs[0]->dimension();

        std::vector<std::pair<const std::vector<int> *, int *>> data = {{&b_, &dim_b}, {&h_, &dim_h}, {&s_, &dim_s}, {&d_, &dim_d}};
        for (auto &pair : data) {
            if (pair.first->size() == 2) {
                *pair.second = (*pair.first)[1] - (*pair.first)[0];
            } else if (pair.first->size() == 1) {
                *pair.second = 1;
            }
        }

        outputs[0]->reshape(dim_b, dim_h, dim_s, dim_d);
        outputs[0]->setDtype(inputs[0]->dtype());
        return ErrorCode::MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        int dim_b = inputs[0]->batch();
        int dim_h = inputs[0]->head();
        int dim_s = inputs[0]->sequence();
        int dim_d = inputs[0]->dimension();
        std::vector<std::pair<std::vector<int>, int *>> data = {{b_, &dim_b}, {h_, &dim_h}, {s_, &dim_s}, {d_, &dim_d}};
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

        if (s_.size() == 2) {
#pragma omp parallel for collapse(1) num_threads(CPUBackend::cpu_threads)
            for (int b = 0; b < inputs[0]->batch(); ++b) {
                memcpy(outputs[0]->hostPtr<float>() + outputs[0]->offset(b, 0, 0, 0),
                       inputs[0]->hostPtr<float>() + inputs[0]->offset(b, 0, s_[0], 0),
                       inputs[0]->head() * (s_[1] - s_[0]) * inputs[0]->dimension() * sizeof(float));
            }
        } else if (s_.size() == 1) {
            int seq_idx = s_[0];
            if (seq_idx < 0) {
                seq_idx = inputs[0]->sequence() + seq_idx;
            }
#pragma omp parallel for collapse(1) num_threads(CPUBackend::cpu_threads)
            for (int b = 0; b < inputs[0]->batch(); ++b) {
                memcpy(outputs[0]->hostPtr<float>() + outputs[0]->offset(b, 0, 0, 0),
                       inputs[0]->hostPtr<float>() + inputs[0]->offset(b, 0, seq_idx, 0),
                       inputs[0]->head() * 1 * inputs[0]->dimension() * sizeof(float));
            }
        } else if (b_.size() == 1) {
            int bth_idx = b_[0];
            if (bth_idx < 0) {
                bth_idx = inputs[0]->batch() + bth_idx;
            }
            memcpy(outputs[0]->hostPtr<float>(),
                   inputs[0]->hostPtr<float>() + inputs[0]->offset(bth_idx, 0, 0, 0),
                   inputs[0]->head() * inputs[0]->sequence() * inputs[0]->dimension() * sizeof(float));
        } else if (b_.size() == 2) {
            assert(b_[1] - b_[0] > 0);
            memcpy(outputs[0]->hostPtr<float>(),
                   inputs[0]->hostPtr<float>() + inputs[0]->offset(b_[0], 0, 0, 0),
                   (b_[1] - b_[0]) * inputs[0]->head() * inputs[0]->sequence() * inputs[0]->dimension() * sizeof(float));
        } else if (d_.size() == 2) {
#pragma omp parallel for collapse(1) num_threads(CPUBackend::cpu_threads)
            for (int b = 0; b < inputs[0]->batch(); ++b) {
                for (int s = 0; s < inputs[0]->sequence(); ++s) {
                    for (int h = 0; h < inputs[0]->head(); ++h) {
                        memcpy(outputs[0]->hostPtr<float>() + outputs[0]->offset(b, h, s, 0),
                               inputs[0]->hostPtr<float>() + inputs[0]->offset(b, h, s, d_[0]),
                               (d_[1] - d_[0]) * sizeof(float));
                    }
                }
            }
        } else if (d_.size() == 1) {
            int seq_idx = d_[0];
            if (seq_idx < 0) {
                seq_idx = inputs[0]->dimension() + seq_idx;
            }
#pragma omp parallel for collapse(1) num_threads(CPUBackend::cpu_threads)
            for (int b = 0; b < inputs[0]->batch(); ++b) {
                for (int s = 0; s < inputs[0]->sequence(); ++s) {
                    for (int h = 0; h < inputs[0]->head(); ++h) {
                        memcpy(outputs[0]->hostPtr<float>() + outputs[0]->offset(b, h, s, 0),
                               inputs[0]->hostPtr<float>() + inputs[0]->offset(b, h, s, seq_idx),
                               sizeof(float));
                    }
                }
            }
        } else {
            std::cout << "[TODO]Tensor.CLip not support!!!!" << std::endl;
        }

        return ErrorCode::MLLM_NO_ERROR;
    }
};

class CPUclipFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // Assumes OpParam is structured to reconstruct the vectors.
        // Example structure: {"b_size": 1, "b_0": 5, "h_size": 0, ...}
        int b_size = op_param.at("b_size");
        int h_size = op_param.at("h_size");
        int s_size = op_param.at("s_size");
        int d_size = op_param.at("d_size");

        std::vector<int> b, h, s, d;
        for (int i = 0; i < b_size; ++i) b.push_back(op_param.at("b_" + std::to_string(i)));
        for (int i = 0; i < h_size; ++i) h.push_back(op_param.at("h_" + std::to_string(i)));
        for (int i = 0; i < s_size; ++i) s.push_back(op_param.at("s_" + std::to_string(i)));
        for (int i = 0; i < d_size; ++i) d.push_back(op_param.at("d_" + std::to_string(i)));

        return new CPUclipFunction(bn, name, threadCount, b, h, s, d);
    }
};

class CPUclipaxisFunction : public Op {
private:
    int thread_count = 4;
    Chl axis_;
    std::vector<int> b_;
    std::vector<int> h_;
    std::vector<int> s_;
    std::vector<int> d_;

public:
    CPUclipaxisFunction(Backend *bn, string name, int threadCount, Chl axis,
                        const std::vector<int> &b, const std::vector<int> &h,
                        const std::vector<int> &s, const std::vector<int> &d) :
        Op(bn, name),
        thread_count(threadCount), axis_(axis), b_(b), h_(h), s_(s), d_(d) {
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        int dim_b = inputs[0]->batch();
        int dim_h = inputs[0]->head();
        int dim_s = inputs[0]->sequence();
        int dim_d = inputs[0]->dimension();
        switch (axis_) {
        case BATCH: {
            std::vector<std::pair<std::vector<int>, int *>> data = {{h_, &dim_h}, {s_, &dim_s}, {d_, &dim_d}};
            for (auto &pair : data) {
                if (!pair.first.empty()) {
                    *pair.second = 1;
                }
            }
            break;
        }
        case HEAD: {
            std::vector<std::pair<std::vector<int>, int *>> data = {{b_, &dim_b}, {s_, &dim_s}, {d_, &dim_d}};
            for (auto &pair : data) {
                if (!pair.first.empty()) {
                    *pair.second = 1;
                }
            }
            break;
        }
        case SEQUENCE: {
            std::vector<std::pair<std::vector<int>, int *>> data = {{b_, &dim_b}, {h_, &dim_h}, {d_, &dim_d}};
            for (auto &pair : data) {
                if (!pair.first.empty()) {
                    *pair.second = 1;
                }
            }
            break;
        }
        case DIMENSION: {
            std::vector<std::pair<std::vector<int>, int *>> data = {{b_, &dim_b}, {h_, &dim_h}, {s_, &dim_s}};
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
        return ErrorCode::MLLM_NO_ERROR;
    }
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        if (axis_ == BATCH) {
            if (!s_.empty()) {
                for (int i = 0; i < s_.size(); ++i) {
                    auto seq_idx = s_[i];
                    memcpy(outputs[0]->hostPtr<char>() + outputs[0]->offset(i, 0, 0, 0),
                           inputs[0]->hostPtr<char>() + inputs[0]->offset(i, 0, seq_idx, 0),
                           inputs[0]->head() * 1 * inputs[0]->dimension() * sizeof(float));
                }
            }
        } else {
            std::cout << "[TODO]Tensor.CLip axis not support!!!!" << std::endl;
        }
        return ErrorCode::MLLM_NO_ERROR;
    }
};

class CPUclipaxisFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        Chl axis = (Chl)op_param.at("axis");
        int b_size = op_param.count("b_size") ? op_param.at("b_size") : 0;
        int h_size = op_param.count("h_size") ? op_param.at("h_size") : 0;
        int s_size = op_param.count("s_size") ? op_param.at("s_size") : 0;
        int d_size = op_param.count("d_size") ? op_param.at("d_size") : 0;

        std::vector<int> b, h, s, d;
        for (int i = 0; i < b_size; ++i) b.push_back(op_param.at("b_" + std::to_string(i)));
        for (int i = 0; i < h_size; ++i) h.push_back(op_param.at("h_" + std::to_string(i)));
        for (int i = 0; i < s_size; ++i) s.push_back(op_param.at("s_" + std::to_string(i)));
        for (int i = 0; i < d_size; ++i) d.push_back(op_param.at("d_" + std::to_string(i)));

        return new CPUclipaxisFunction(bn, name, threadCount, axis, b, h, s, d);
    }
};

class CPUcliptensorFunction : public Op {
private:
    int thread_count = 4;
    Chl dim_;

public:
    CPUcliptensorFunction(Backend *bn, string name, int threadCount, Chl dim) :
        Op(bn, name), thread_count(threadCount), dim_(dim) {
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        if (dim_ == SEQUENCE) {
            int new_seq = inputs[1]->dimension();
            outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), new_seq, inputs[0]->dimension());
        } else if (dim_ == DIMENSION) {
            int new_dim = inputs[1]->dimension();
            outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), new_dim);
        } else {
            std::cout << "[TODO]Tensor.Clip tensor not support!!!!" << std::endl;
        }
        outputs[0]->setDtype(inputs[0]->dtype());
        return ErrorCode::MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        if (dim_ == SEQUENCE) {
            if (inputs[0]->ctype() == BHDS) {
                outputs[0]->chls() = inputs[0]->chls();
                outputs[0]->setCtype(BHDS);
                int new_seq = inputs[1]->dimension();
                if (outputs[0]->sequence() == 0 || outputs[0]->shape().empty()
                    || new_seq != outputs[0]->sequence()) {
                    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), new_seq, inputs[0]->dimension());
                    outputs[0]->alloc();
                }

#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
                for (int b = 0; b < inputs[0]->batch(); ++b) {
                    for (int d = 0; d < inputs[0]->dimension(); ++d) {
                        for (int s = 0; s < new_seq; ++s) {
                            auto selected_idx = (int)inputs[1]->dataAt<float>(0, 0, 0, s);
                            outputs[0]->setDataAt<float>(b, 0, s, d,
                                                         inputs[0]->dataAt<float>(b, 0, selected_idx, d));
                        }
                    }
                }
                return MLLM_NO_ERROR;
            }
            int new_seq = inputs[1]->dimension();
            if (outputs[0]->sequence() == 0 || outputs[0]->shape().empty()
                || new_seq != outputs[0]->sequence()) {
                outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), new_seq, inputs[0]->dimension());
                outputs[0]->alloc();
            }
            for (int b = 0; b < inputs[0]->batch(); ++b) {
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
                for (int s = 0; s < inputs[1]->dimension(); ++s) {
                    auto selected_idx = (int)inputs[1]->dataAt<float>(0, 0, 0, s);
                    memcpy(outputs[0]->ptrAt<float>(b, 0, s, 0),
                           inputs[0]->ptrAt<float>(b, 0, selected_idx, 0),
                           inputs[0]->head() * inputs[0]->dimension() * sizeof(float));
                }
            }
        } else if (dim_ == DIMENSION) {
            int new_seq = inputs[1]->dimension();
            if (outputs[0]->sequence() == 0 || outputs[0]->shape().empty()
                || new_seq != outputs[0]->sequence()) {
                outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), new_seq);
                outputs[0]->alloc();
            }
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
            for (int b = 0; b < inputs[0]->batch(); ++b) {
                for (int s = 0; s < inputs[0]->sequence(); ++s) {
                    for (int d = 0; d < inputs[1]->dimension(); ++d) {
                        auto selected_idx = (int)inputs[1]->dataAt<float>(0, 0, 0, d);
                        outputs[0]->setDataAt<float>(b, 0, s, d,
                                                     inputs[0]->dataAt<float>(b, 0, s, selected_idx));
                    }
                }
            }
        } else {
            std::cout << "[TODO]Tensor.CLip not support!!!!" << std::endl;
        }
        return ErrorCode::MLLM_NO_ERROR;
    }
};

class CPUcliptensorFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        Chl dim = (Chl)op_param.at("dim");
        return new CPUcliptensorFunction(bn, name, threadCount, dim);
    }
};

} // namespace mllm
#endif // CPUCLIPFUNC_HPP