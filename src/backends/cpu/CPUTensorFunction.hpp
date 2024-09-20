//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUTENSORFUNCTION_HPP
#define CPUTENSORFUNCTION_HPP
#include "CPUBackend.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "compute/Matmul.hpp"
#include "compute/Arithmetic.hpp"

// #include <Layer.hpp>
#include <iostream>
#include <vector>

namespace mllm {
class Tensor;

class CPUmmFunction : public TensorFunction {
    static void tranTensorChl(Tensor &input) {
        assert(input.ctype() == BSHD);
        auto b = input.batch();
        auto h = input.head();
        auto d = input.dimension();
        auto s = input.sequence();
        auto ori_seq_idx = input.chls()[SEQUENCE];
        auto ori_head_idx = input.chls()[HEAD];
        auto ori_dim_idx = input.chls()[DIMENSION];
        input.chls()[HEAD] = ori_seq_idx;
        input.chls()[DIMENSION] = ori_head_idx;
        input.chls()[SEQUENCE] = ori_dim_idx;
        input.changeCtype();
        input.reshape(b, h, s, d);
        input.transed() = true;
        input.undiffusion() = false;
        // if no TENSOR_STATIC_SHAPED
        if (input.masterTensor() != nullptr) {
            auto b = input.masterTensor()->batch();
            auto h = input.masterTensor()->head();
            auto d = input.masterTensor()->dimension();
            auto s = input.masterTensor()->sequence();
            input.masterTensor()->chls() = input.chls();
            input.masterTensor()->changeCtype();
            input.masterTensor()->reshape(b, h, s, d);
            for (auto child : input.masterTensor()->childTensors()) {
                auto b = child->batch();
                auto h = child->head();
                auto d = child->dimension();
                auto s = child->sequence();
                child->chls() = input.chls();
                child->changeCtype();
                child->reshape(b, h, s, d);
            }
        }
    }

public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        if (inputs[1]->chls()[SEQUENCE] != 3) {
            tranTensorChl(*inputs[1]);
        }
        assert(inputs[0]->dimension() == inputs[1]->sequence());
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[1]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        bool isSame = std::equal(inputs[0]->chls().begin(), inputs[0]->chls().end(), inputs[1]->chls().begin());
        assert(inputs[0]->dtype() == MLLM_TYPE_F32);
        mat_mul(inputs[0], inputs[1], outputs[0], false, nullptr, false, isSame, CPUBackend::cpu_threads);
        /*
        switch (inputs[1]->dtype()) {
        case MLLM_TYPE_F32: {
            mat_mul_fp32(inputs[0], inputs[1], outputs[0], false, nullptr, false, isSame, CPUBackend::cpu_threads);
            break;
        }
        case MLLM_TYPE_F16: {
            mat_mul_fp32_fp16(inputs[0], inputs[1], outputs[0], false, nullptr, false, isSame, CPUBackend::cpu_threads);
            break;
        }
        default:
            break;
        }
        */
    }
};

class CPUnormFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        int L_n = (int)args[0];
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        int L_n = (int)args[0];
        for (int h = 0; h < inputs[0]->head(); h++) {
            for (int n = 0; n < inputs[0]->batch(); n++) {
                for (int s = 0; s < inputs[0]->sequence(); s++) {
                    if (L_n == 2) {
                        float sum_of_squares = 0.0f;
                        for (int d = 0; d < inputs[0]->dimension(); ++d) {
                            sum_of_squares += inputs[0]->dataAt<float>(n, h, s, d) * inputs[0]->dataAt<float>(n, h, s, d);
                        }
                        float l2_norm = std::sqrt(sum_of_squares);
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
                        for (int d = 0; d < inputs[0]->dimension(); d++) {
                            outputs[0]->setDataAt<float>(n, h, s, d, l2_norm);
                        }
                    } else {
                        float sum_of_abs_values = 0.0f;
                        for (int d = 0; d < inputs[0]->dimension(); ++d) {
                            sum_of_abs_values += std::abs(inputs[0]->dataAt<float>(n, h, s, d));
                        }
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
                        for (int d = 0; d < inputs[0]->dimension(); d++) {
                            outputs[0]->setDataAt<float>(n, h, s, d, sum_of_abs_values);
                        }
                    }
                }
            }
        }
    }
};
/*
class CPUbinaryFunction {
public:
    template <typename Func>
    void setup(Tensor *input, Tensor *output, Func operation, float data) {
        output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
        output->setDtype(input->dtype());
        output->alloc();
    }
    template <typename Func>
    void execute(Tensor *input, Tensor *output, Func operation, float data) {
        if (input->masterTensor() == nullptr && output->masterTensor() == nullptr && input->ctype() == output->ctype()) {
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
            for (int is = 0; is < input->batch() * input->head() * input->sequence() * input->dimension(); ++is) {
                output->hostPtr<float>()[is] = operation(input->hostPtr<float>()[is], data);
            }
        } else {
            for (int n = 0; n < input->batch(); ++n) {
                for (int c = 0; c < input->head(); ++c) {
                    for (int h = 0; h < input->sequence(); ++h) {
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
                        for (int w = 0; w < input->dimension(); ++w) {
                            output->ptrAt<float>(n, c, h, w)[0] =
                                operation(input->ptrAt<float>(n, c, h, w)[0],
                                          data);
                        }
                    }
                }
            }
        }
    }
};
*/
class CPUaddFunction : public TensorFunction { //, public CPUbinaryFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        // float data = (float)args[0];
        auto input = inputs[0];
        auto output = outputs[0];
        output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
        output->setDtype(input->dtype());
        output->alloc();
        // CPUbinaryFunction::setup( inputs[0], outputs[0], std::plus<float>(), data);
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        float data = (float)args[0];
        auto input = inputs[0];
        auto output = outputs[0];
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
        for (int n = 0; n < input->batch(); ++n) {
            for (int c = 0; c < input->head(); ++c) {
                for (int h = 0; h < input->sequence(); ++h) {
                    mllm_add_fp32(input->ptrAt<float>(n, c, h, 0), data,
                                  outputs[0]->ptrAt<float>(n, c, h, 0), input->dimension());
                }
            }
        }
        // CPUbinaryFunction::execute( inputs[0], outputs[0], std::plus<float>(), data);
    }
};
class CPUsubFunction : public TensorFunction { //, public CPUbinaryFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        // float data = (float)args[0];
        auto input = inputs[0];
        auto output = outputs[0];
        output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
        output->setDtype(input->dtype());
        output->alloc();
        // CPUbinaryFunction::setup( inputs[0], outputs[0], std::minus<float>(), data);
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        float data = (float)args[0];
        auto input = inputs[0];
        auto output = outputs[0];
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
        for (int n = 0; n < input->batch(); ++n) {
            for (int c = 0; c < input->head(); ++c) {
                for (int h = 0; h < input->sequence(); ++h) {
                    mllm_sub_fp32(input->ptrAt<float>(n, c, h, 0), data,
                                  outputs[0]->ptrAt<float>(n, c, h, 0), input->dimension());
                }
            }
        }
        // CPUbinaryFunction::execute( inputs[0], outputs[0], std::minus<float>(), data);
    }
};
class CPUmulFunction : public TensorFunction { //, public CPUbinaryFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        // float data = (float)args[0];
        auto input = inputs[0];
        auto output = outputs[0];
        output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
        output->setDtype(input->dtype());
        output->alloc();
        // CPUbinaryFunction::setup( inputs[0], outputs[0], std::multiplies<float>(), data);
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        float data = (float)args[0];
        auto input = inputs[0];
        auto output = outputs[0];
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
        for (int n = 0; n < input->batch(); ++n) {
            for (int c = 0; c < input->head(); ++c) {
                for (int h = 0; h < input->sequence(); ++h) {
                    mllm_mul_fp32(input->ptrAt<float>(n, c, h, 0), data,
                                  outputs[0]->ptrAt<float>(n, c, h, 0), input->dimension());
                }
            }
        }
        // CPUbinaryFunction::execute( inputs[0], outputs[0], std::multiplies<float>(), data);
    }
};
class CPUdivFunction : public TensorFunction { //, public CPUbinaryFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        // float data = (float)args[0];
        auto input = inputs[0];
        auto output = outputs[0];
        output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
        output->setDtype(input->dtype());
        output->alloc();
        // CPUbinaryFunction::setup( inputs[0], outputs[0], std::divides<float>(), data);
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        float data = (float)args[0];
        auto input = inputs[0];
        auto output = outputs[0];
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
        for (int n = 0; n < input->batch(); ++n) {
            for (int c = 0; c < input->head(); ++c) {
                for (int h = 0; h < input->sequence(); ++h) {
                    mllm_div_fp32(input->ptrAt<float>(n, c, h, 0), data,
                                  outputs[0]->ptrAt<float>(n, c, h, 0), input->dimension());
                }
            }
        }
        // CPUbinaryFunction::execute( inputs[0], outputs[0], std::divides<float>(), data);
    }
};
/*
class CPUbinaryTwoFunction {
public:
    template <typename Func>
    void setup(Tensor *input0,  Tensor *input1, Tensor *output,   Func operation) {
        output->reshape(std::max(input0->batch(), input1->batch()),
                        input0->head(), input0->sequence(), input0->dimension());
        output->setDtype(input0->dtype());
        output->alloc();
    }
    template <typename Func>
    void execute(Tensor *input0, Tensor *input1, Tensor *output,   Func operation) {
        int batch_ = std::max(input0->batch(), input1->batch());
        if (input0->masterTensor() == nullptr && output->masterTensor() == nullptr && input0->ctype() == output->ctype()) {
            for (int n = 0; n < batch_; ++n) {
                auto n_0 = std::min(n, input0->batch() - 1);
                auto n_1 = std::min(n, input1->batch() - 1);
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
                for (int is = 0; is < input0->head() * input0->sequence() * input0->dimension(); ++is) {
                    output->ptrAt<float>(n, 0, 0, 0)[is] =
                        operation(input0->ptrAt<float>(n_0, 0, 0, 0)[is],
                                  input1->ptrAt<float>(n_1, 0, 0, 0)[is]);
                }
            }
        } else {
            for (int n = 0; n < batch_; ++n) {
                auto n_0 = std::min(n, input0->batch() - 1);
                auto n_1 = std::min(n, input1->batch() - 1);
                for (int c = 0; c < input0->head(); ++c) {
                    for (int h = 0; h < input0->sequence(); ++h) {
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
                        for (int w = 0; w < input0->dimension(); ++w) {
                            output->ptrAt<float>(n, c, h, w)[0] =
                                operation(input0->ptrAt<float>(n_0, c, h, w)[0],
                                          input1->ptrAt<float>(n_1, c, h, w)[0]);
                        }
                    }
                }
            }
        }
    }
};
*/
class CPUaddTwoFunction : public TensorFunction { //, public CPUbinaryTwoFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        outputs[0]->reshape(std::max(inputs[0]->batch(), inputs[1]->batch()),
                            inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
        // CPUbinaryTwoFunction::setup( inputs[0], inputs[1], outputs[0], std::plus<float>());
    };
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        auto input0 = inputs[0];
        auto input1 = inputs[1];
        int batch_ = std::max(input0->batch(), input1->batch());
        for (int n = 0; n < batch_; ++n) {
            auto n_0 = std::min(n, input0->batch() - 1);
            auto n_1 = std::min(n, input1->batch() - 1);
#pragma omp parallel for collapse(2) num_threads(CPUBackend::cpu_threads)
            for (int c = 0; c < input0->head(); ++c) {
                for (int h = 0; h < input0->sequence(); ++h) {
                    mllm_add_fp32(input0->ptrAt<float>(n_0, c, h, 0), input1->ptrAt<float>(n_0, c, h, 0),
                                  outputs[0]->ptrAt<float>(n_0, c, h, 0), input0->dimension());
                }
            }
        }
        // CPUbinaryTwoFunction::execute( inputs[0], inputs[1], outputs[0], std::plus<float>());
    };
};
class CPUsubTwoFunction : public TensorFunction { //, public CPUbinaryTwoFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        outputs[0]->reshape(std::max(inputs[0]->batch(), inputs[1]->batch()),
                            inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
        // CPUbinaryTwoFunction::setup( inputs[0], inputs[1], outputs[0], std::minus<float>());
    };
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        auto input0 = inputs[0];
        auto input1 = inputs[1];
        int batch_ = std::max(input0->batch(), input1->batch());
        for (int n = 0; n < batch_; ++n) {
            auto n_0 = std::min(n, input0->batch() - 1);
            auto n_1 = std::min(n, input1->batch() - 1);
#pragma omp parallel for collapse(2) num_threads(CPUBackend::cpu_threads)
            for (int c = 0; c < input0->head(); ++c) {
                for (int h = 0; h < input0->sequence(); ++h) {
                    mllm_sub_fp32(input0->ptrAt<float>(n_0, c, h, 0), input1->ptrAt<float>(n_0, c, h, 0),
                                  outputs[0]->ptrAt<float>(n_0, c, h, 0), input0->dimension());
                }
            }
        }
        // CPUbinaryTwoFunction::execute( inputs[0], inputs[1], outputs[0], std::minus<float>());
    };
};
class CPUmulTwoFunction : public TensorFunction { //, public CPUbinaryTwoFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        outputs[0]->reshape(std::max(inputs[0]->batch(), inputs[1]->batch()),
                            inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
        // CPUbinaryTwoFunction::setup( inputs[0], inputs[1], outputs[0], std::multiplies<float>());
    };
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        auto input0 = inputs[0];
        auto input1 = inputs[1];
        int batch_ = std::max(input0->batch(), input1->batch());
        for (int n = 0; n < batch_; ++n) {
            auto n_0 = std::min(n, input0->batch() - 1);
            auto n_1 = std::min(n, input1->batch() - 1);
#pragma omp parallel for collapse(2) num_threads(CPUBackend::cpu_threads)
            for (int c = 0; c < input0->head(); ++c) {
                for (int h = 0; h < input0->sequence(); ++h) {
                    mllm_mul_fp32(input0->ptrAt<float>(n_0, c, h, 0), input1->ptrAt<float>(n_0, c, h, 0),
                                  outputs[0]->ptrAt<float>(n_0, c, h, 0), input0->dimension());
                }
            }
        }
        //  CPUbinaryTwoFunction::execute( inputs[0], inputs[1], outputs[0], std::multiplies<float>());
    };
};
class CPUdivTwoFunction : public TensorFunction { //, public CPUbinaryTwoFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        outputs[0]->reshape(std::max(inputs[0]->batch(), inputs[1]->batch()),
                            inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
        // CPUbinaryTwoFunction::setup( inputs[0], inputs[1], outputs[0], std::divides<float>());
    };
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        auto input0 = inputs[0];
        auto input1 = inputs[1];
        int batch_ = std::max(input0->batch(), input1->batch());
        for (int n = 0; n < batch_; ++n) {
            auto n_0 = std::min(n, input0->batch() - 1);
            auto n_1 = std::min(n, input1->batch() - 1);
#pragma omp parallel for collapse(2) num_threads(CPUBackend::cpu_threads)
            for (int c = 0; c < input0->head(); ++c) {
                for (int h = 0; h < input0->sequence(); ++h) {
                    mllm_div_fp32(input0->ptrAt<float>(n_0, c, h, 0), input1->ptrAt<float>(n_0, c, h, 0),
                                  outputs[0]->ptrAt<float>(n_0, c, h, 0), input0->dimension());
                }
            }
        }
        // CPUbinaryTwoFunction::execute( inputs[0], inputs[1], outputs[0], std::divides<float>());
    };
};

class CPUmeanFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        Chl axis = (Chl)args[0];
        int batch = inputs[0]->batch();
        int head = inputs[0]->head();
        int sequence = inputs[0]->sequence();
        int dimension = inputs[0]->dimension();
        switch (axis) {
        case BATCH:
            batch = 1;
            break;
        case HEAD:
            head = 1;
            break;
        case SEQUENCE:
            sequence = 1;
            break;
        case DIMENSION:
            dimension = 1;
            break;
        default:
            break;
        }
        outputs[0]->reshape(batch, head, sequence, dimension);
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        Chl axis = (Chl)args[0];
        int batch = inputs[0]->batch();
        int dim = inputs[0]->dimension();
        int seq = inputs[0]->sequence();
        int head = inputs[0]->head();
        switch (axis) {
        case BATCH: {
            for (int h = 0; h < head; h++) {
                for (int s = 0; s < seq; ++s) {
                    for (int d = 0; d < dim; d++) {
                        float sum = 0.0f;
                        for (int n = 0; n < batch; n++) {
                            sum += inputs[0]->dataAt<float>(n, h, s, d);
                        }
                        outputs[0]->setDataAt<float>(0, h, s, d, sum / seq);
                    }
                }
            }
            break;
        }
        case HEAD: {
            for (int n = 0; n < batch; n++) {
                for (int s = 0; s < seq; ++s) {
                    for (int d = 0; d < dim; d++) {
                        float sum = 0.0f;
                        for (int h = 0; h < head; h++) {
                            sum += inputs[0]->dataAt<float>(n, h, s, d);
                        }
                        outputs[0]->setDataAt<float>(n, 0, s, d, sum / seq);
                    }
                }
            }
            break;
        }
        case SEQUENCE: {
            for (int n = 0; n < batch; n++) {
                for (int h = 0; h < head; h++) {
                    for (int d = 0; d < dim; d++) {
                        float sum = 0.0f;
                        for (int s = 0; s < seq; ++s) {
                            sum += inputs[0]->dataAt<float>(n, h, s, d);
                        }
                        outputs[0]->setDataAt<float>(n, h, 0, d, sum / seq);
                    }
                }
            }
            break;
        }
        case DIMENSION: {
            for (int n = 0; n < batch; n++) {
                for (int h = 0; h < head; h++) {
                    for (int s = 0; s < seq; s++) {
                        float sum = 0.0f;
                        for (int d = 0; d < inputs[0]->dimension(); ++d) {
                            sum += inputs[0]->dataAt<float>(n, h, s, d);
                        }
                        outputs[0]->setDataAt<float>(n, h, s, 0, sum / inputs[0]->dimension());
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

class CPUviewFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        int b = (int)args[0];
        int h = (int)args[1];
        int s = (int)args[2];
        int d = (int)args[3];
        int dim_b = inputs[0]->batch();
        int dim_h = inputs[0]->head();
        int dim_s = inputs[0]->sequence();
        int dim_d = inputs[0]->dimension();
        if (b == -1 && h != -1 && s == -1 && d != -1) { // head & dimension
            if (h != ANYDIM && d != ANYDIM) {
                assert(inputs[0]->dimension() * inputs[0]->head() == h * d);
                dim_h = h;
                dim_d = d;
            } else if (h != ANYDIM) {
                dim_h = h;
                dim_d = inputs[0]->dimension() * inputs[0]->head() / h;
            } else if (d != ANYDIM) {
                dim_h = inputs[0]->dimension() * inputs[0]->head() / d;
                dim_d = d;
            } else {
                std::cout << "[TODO]Tensor.View not support!!!!" << std::endl;
            }
        } else if (b == -1 && h != -1 && s != -1 && d == -1) { // head & sequence
            if (h != ANYDIM && s != ANYDIM) {
                assert(inputs[0]->sequence() * inputs[0]->head() == h * s);
                dim_h = h;
                dim_s = s;
            } else if (h != ANYDIM) {
                dim_h = h;
                dim_s = inputs[0]->sequence() * inputs[0]->head() / h;
            } else if (s != ANYDIM) {
                dim_h = inputs[0]->sequence() * inputs[0]->head() / s;
                dim_s = s;
            } else {
                std::cout << "[TODO]Tensor.View not support!!!!" << std::endl;
            }
        } else if (b != -1 && h == -1 && s != -1 && d == -1) { // batch & sequence
            if (b != ANYDIM && s != ANYDIM) {
                assert(inputs[0]->sequence() * inputs[0]->batch() == b * s);
                dim_b = b;
                dim_s = s;
            } else if (b != ANYDIM) {
                dim_b = b;
                dim_s = inputs[0]->sequence() * inputs[0]->batch() / b;
            } else if (s != ANYDIM) {
                dim_b = inputs[0]->sequence() * inputs[0]->batch() / s;
                dim_s = s;
            } else {
                std::cout << "[TODO]Tensor.View not support!!!!" << std::endl;
            }
        } else {
            std::cout << "[TODO]Tensor.View not support!!!!" << std::endl;
        }
        outputs[0]->reshape(dim_b, dim_h, dim_s, dim_d);
        if ((b == -1 && s == -1 && inputs[0]->ctype() != BCTHW)   // head & dimension
            || (b == -1 && d == -1 && inputs[0]->ctype() == BSHD) // head & sequence
            || (h == -1 && d == -1 && inputs[0]->ctype() == BSHD) // batch & sequence
        ) {
            if (inputs[0]->masterTensor() == nullptr) {
                inputs[0]->free();
            }
            outputs[0]->setDtype(inputs[0]->dtype());
            outputs[0]->alloc();
            inputs[0]->deepCopyFrom(outputs[0], false);
        } else {
            std::cout << "[TODO]Tensor.View not support!!!!" << std::endl;
        }
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
    }
};

class CPUflattenFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        Chl axis_start = (Chl)args[0];
        Chl axis_end = (Chl)args[1];
        int dim_b = inputs[0]->batch();
        int dim_h = 0;
        int dim_s = 0;
        int dim_d = 0;
        if (inputs[0]->shape().size() == 4) {
            dim_h = inputs[0]->head();
            dim_s = inputs[0]->sequence();
            dim_d = inputs[0]->dimension();
            if (axis_start == BATCH & axis_end == SEQUENCE) {
                dim_b = 1;
                dim_s = inputs[0]->sequence() * inputs[0]->batch();
            } else if (axis_start == HEAD & axis_end == SEQUENCE) {
                dim_h = 1;
                dim_s = inputs[0]->sequence() * inputs[0]->head();
            } else if (axis_start == HEAD & axis_end == DIMENSION) {
                dim_h = 1;
                dim_d = inputs[0]->dimension() * inputs[0]->head();
            } else {
                std::cout << "ERROR:  flatten  " << axis_start << "&" << axis_end << std::endl;
            }
        } else if (inputs[0]->shape().size() == 5) {
            if (axis_start == CHANNLE & axis_end == HEIGHT) {
                dim_h = 1;
                dim_s = inputs[0]->channel() * inputs[0]->height() * inputs[0]->time();
                dim_d = inputs[0]->width();
            } else if (axis_start == HEIGHT & axis_end == CHANNLE) {
                dim_h = 1;
                dim_s = inputs[0]->channel() * inputs[0]->height() * inputs[0]->width();
                dim_d = inputs[0]->time();
            }
        }
        assert(dim_d + dim_s + dim_h > 0);
        outputs[0]->reshape(dim_b, dim_h, dim_s, dim_d);
        if ((axis_start == TIME & axis_end == WIDTH && inputs[0]->ctype() == BCTHW)
            || (axis_start == CHANNLE & axis_end == HEIGHT && inputs[0]->ctype() == BWCTH)
            || (axis_start == HEIGHT & axis_end == CHANNLE && inputs[0]->ctype() == BTHWC)
            || (axis_start == BATCH & axis_end == SEQUENCE && inputs[0]->ctype() != BCTHW)
            || (axis_start == HEAD & axis_end == SEQUENCE && inputs[0]->ctype() == BSHD)
            || (axis_start == HEAD & axis_end == SEQUENCE && inputs[0]->ctype() == BHDS)
            || (axis_start == HEAD & axis_end == SEQUENCE && inputs[0]->ctype() == BDHS)
            || (axis_start == HEAD & axis_end == DIMENSION && inputs[0]->ctype() == BSHD)
            || (axis_start == HEAD & axis_end == DIMENSION && inputs[0]->ctype() == BHDS)
            || (axis_start == HEAD & axis_end == SEQUENCE && inputs[0]->ctype() == BDSH)) {
            if (inputs[0]->masterTensor() == nullptr) {
                inputs[0]->free();
            }
            outputs[0]->setDtype(inputs[0]->dtype());
            outputs[0]->alloc();
            inputs[0]->deepCopyFrom(outputs[0], false);
        } else {
            std::cout << "[TODO]Tensor.Flatten not support!!!!" << std::endl;
        }
    }

    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
    }
};
class CPUtransposeFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        vector<std::pair<Chl, Chl>> axiss;
        for (int i = 0; i < args.size(); i += 2) {
            axiss.push_back({(Chl)args[i], (Chl)args[i + 1]});
        }
        if (outputs[0]->count() <= 0 || outputs[0]->shape() != inputs[0]->shape()) {
            outputs[0]->transCopyShape(inputs[0]->shape());
            std::map<Chl, int> origin_chls = {{BATCH, 0}, {SEQUENCE, 1}, {HEAD, 2}, {DIMENSION, 3}, {CHANNLE, 1}, {TIME, 2}, {HEIGHT, 3}, {WIDTH, 4}};
            if (std::equal(outputs[0]->chls().begin(), outputs[0]->chls().end(), origin_chls.begin())) {
                outputs[0]->chls() = inputs[0]->chls();
                for (auto axis : axiss) {
                    auto axis0 = axis.first;
                    auto axis1 = axis.second;
                    auto ori_0_idx = outputs[0]->chls()[axis0];
                    auto ori_1_idx = outputs[0]->chls()[axis1];
                    outputs[0]->chls()[axis0] = ori_1_idx;
                    outputs[0]->chls()[axis1] = ori_0_idx;
                }
                outputs[0]->changeCtype(inputs[0]->shape().size());
                outputs[0]->undiffusion() = true;
            }
            if (inputs[0]->masterTensor() != nullptr) {
                if (outputs[0]->masterTensor() == nullptr) {
                    outputs[0]->setDtype(inputs[0]->dtype());
                    outputs[0]->deepCopyFrom(inputs[0], false);
                }
            } else {
                if (inputs[0]->masterTensor() == nullptr) {
                    inputs[0]->free();
                }
                outputs[0]->setDtype(inputs[0]->dtype());
                outputs[0]->alloc();
                // inputs[0]->undiffusion() = true;
                inputs[0]->setUndiffusion(true);
                inputs[0]->deepCopyFrom(outputs[0], false);
                outputs[0]->transFrom() = axiss;
            }
        }
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
    }
};

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
        if (s.size() == 2) {
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
            for (int b = 0; b < inputs[0]->batch(); ++b) {
                memcpy(outputs[0]->hostPtr<float>() + outputs[0]->offset(b, 0, 0, 0),
                       inputs[0]->hostPtr<float>() + inputs[0]->offset(b, 0, seq_idx, 0),
                       inputs[0]->head() * 1 * inputs[0]->dimension() * sizeof(float));
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

class CPUcatFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        Chl axis = (Chl)args[0];
        int expd_batch_ = inputs[0]->batch();
        for (int ii = 0; ii < inputs.size(); ++ii) {
            auto input = inputs[ii];
            if (input->batch() > expd_batch_) {
                expd_batch_ = input->batch();
            }
        }
        int dim_b = expd_batch_;
        int dim_h = inputs[0]->head();
        int dim_s = inputs[0]->sequence();
        int dim_d = inputs[0]->dimension();
        int sizes[] = {0, 0, 0, 0};
        Chl axes[] = {BATCH, HEAD, SEQUENCE, DIMENSION};
        int *dims[] = {&dim_b, &dim_h, &dim_s, &dim_d};
        for (int i = 0; i < 4; i++) {
            if (axis == axes[i]) {
                for (auto input : inputs) {
                    sizes[i] += (i == 0) ? input->batch() : (i == 1) ? input->head() :
                                                        (i == 2)     ? input->sequence() :
                                                                       input->dimension();
                }
                *dims[i] = sizes[i];
                break;
            }
        }
        outputs[0]->reshape(dim_b, dim_h, dim_s, dim_d);
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
        if (axis == SEQUENCE && inputs[0]->head() != 1) {
            int cbatch = 0;
            int chead = 0;
            int cseq = 0;
            int cdim = 0;
            for (int idx = 0; idx < inputs.size(); idx++) {
                if (inputs[idx]->masterTensor() == nullptr) {
                    inputs[idx]->free();
                }
                if (idx > 0) {
                    cseq += inputs[idx - 1]->sequence();
                }
                inputs[idx]->deepCopyFrom(outputs[0], false, {cbatch, chead, cseq, cdim}); // b,h,s,d
            }
        } else if (axis == DIMENSION && inputs[0]->head() != 1) {
            int cbatch = 0;
            int chead = 0;
            int cseq = 0;
            int cdim = 0;
            for (int idx = 0; idx < inputs.size(); idx++) {
                if (inputs[idx]->masterTensor() == nullptr) {
                    inputs[idx]->free();
                }
                if (idx > 0) {
                    cdim += inputs[idx - 1]->dimension();
                }
                int tmp_agg_idx;
                if (inputs[idx]->deaggregatedTensor() != nullptr) {
                    for (int t = 0; t < inputs[idx]->deaggregatedTensor()->aggregatedTensors().size(); t++) {
                        if (inputs[idx]->deaggregatedTensor()->aggregatedTensors()[t].get() == inputs[idx]) {
                            tmp_agg_idx = t;
                            continue;
                        }
                    }
                }
                inputs[idx]->deepCopyFrom(outputs[0], false, {cbatch, chead, cseq, cdim}); // b,h,s,d
                if (inputs[idx]->deaggregatedTensor() != nullptr) {
                    vector<shared_ptr<Tensor>> shared_outputs = {};
                    for (int t = 0; t < inputs[idx]->deaggregatedTensor()->aggregatedTensors().size(); t++) {
                        if (t == tmp_agg_idx) {
                            inputs[idx]->deaggregatedTensor()->aggregatedTensors()[t] =
                                std::shared_ptr<Tensor>(inputs[idx], [](Tensor *) {});
                        }
                    }
                }
            }
        }
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        Chl axis = (Chl)args[0];
        int expd_batch_ = inputs[0]->batch();
        int expd_batch_input_idx = 0;
        for (int ii = 0; ii < inputs.size(); ++ii) {
            auto input = inputs[ii];
            if (input->batch() > expd_batch_) {
                expd_batch_ = input->batch();
                expd_batch_input_idx = ii;
            }
        }
        if (axis == BATCH) {
            for (int n = 0; n < inputs.size(); ++n) {
                auto copysize = inputs[0]->batch() * inputs[0]->head() * inputs[0]->sequence() * inputs[0]->dimension();
                memcpy(outputs[0]->ptrAt<float>(n * inputs[0]->batch(), 0, 0, 0),
                       inputs[n]->ptrAt<float>(0, 0, 0, 0),
                       sizeof(float) * copysize);
            }
        } else if (axis == DIMENSION) {
            for (int n = 0; n < expd_batch_; ++n) {
                for (int c = 0; c < inputs[0]->head(); ++c) {
                    for (int h = 0; h < inputs[0]->sequence(); ++h) {
                        int w = 0;
                        for (int idx = 0; idx < inputs.size(); idx++) {
                            int dim_size = inputs[idx]->dimension();
                            auto n_ = n;
                            if (idx != expd_batch_input_idx) {
                                n_ = 0;
                            }
                            memcpy(outputs[0]->ptrAt<float>(n, c, h, w),
                                   inputs[idx]->ptrAt<float>(n_, c, h, 0),
                                   sizeof(float) * (dim_size));
                            w += dim_size;
                        }
                    }
                }
            }
        } else if ((axis == SEQUENCE) && inputs[0]->head() != 1) {
        } else if ((axis == SEQUENCE) && inputs[0]->head() == 1) {
            for (int n = 0; n < expd_batch_; ++n) {
                int h = 0;
                for (int idx = 0; idx < inputs.size(); idx++) {
                    auto n_ = n;
                    if (idx != expd_batch_input_idx) {
                        n_ = 0;
                    }
                    memcpy(outputs[0]->ptrAt<float>(n, 0, h, 0),
                           inputs[idx]->ptrAt<float>(n_, 0, 0, 0),
                           sizeof(float) * (inputs[idx]->sequence() * inputs[idx]->dimension()));
                    h += inputs[idx]->sequence();
                }
            }
        }
    }
};

class CPUwhereFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        float value = args[0];
        Chl axis = (Chl)args[1];
        vector<float> b_vec = {};
        vector<float> s_vec = {};
        vector<float> h_vec = {};
        vector<float> d_vec = {};
#pragma omp parallel for collapse(4) num_threads(CPUBackend::cpu_threads)
        for (int b = 0; b < inputs[0]->batch(); b++) {
            for (auto s = 0; s < inputs[0]->sequence(); s++) {
                for (auto h = 0; h < inputs[0]->head(); h++) {
                    for (auto d = 0; d < inputs[0]->dimension(); d++) {
                        if (inputs[0]->dataAt<float>(b, h, h, s) == value) {
                            b_vec.push_back(b);
                            s_vec.push_back(s);
                            h_vec.push_back(h);
                            d_vec.push_back(d);
                        }
                    }
                }
            }
        }
        int num = b_vec.size();
        if ((int)axis == -1) {
            outputs[0]->reshape(1, 1, 4, num);
            outputs[0]->setDtype(inputs[0]->dtype());
            outputs[0]->alloc();
            for (int i = 0; i < 4; ++i) {
                auto dest_ptr = outputs[0]->hostPtr<float>() + outputs[0]->offset(0, 0, i, 0);
                switch (i) {
                case 0:
                    memcpy(dest_ptr, b_vec.data(), num * sizeof(float));
                    break;
                case 1:
                    memcpy(dest_ptr, h_vec.data(), num * sizeof(float));
                    break;
                case 2:
                    memcpy(dest_ptr, s_vec.data(), num * sizeof(float));
                    break;
                case 3:
                    memcpy(dest_ptr, d_vec.data(), num * sizeof(float));
                    break;
                default:
                    break;
                }
            }
        } else {
            outputs[0]->reshape(1, 1, 1, num);
            outputs[0]->setDtype(inputs[0]->dtype());
            outputs[0]->alloc();
            auto dest_ptr = outputs[0]->hostPtr<float>();
            switch (axis) {
            case BATCH:
                memcpy(dest_ptr, b_vec.data(), num * sizeof(float));
                break;
            case HEAD:
                memcpy(dest_ptr, h_vec.data(), num * sizeof(float));
                break;
            case SEQUENCE:
                memcpy(dest_ptr, s_vec.data(), num * sizeof(float));
                break;
            case DIMENSION:
                memcpy(dest_ptr, d_vec.data(), num * sizeof(float));
                break;
            default:
                break;
            }
        }
    }
};

class CPURangeFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        int start = (int)args[0];
        int end = (int)args[1];
        outputs[0]->reshape(1, 1, end - start, 1);
        outputs[0]->setDtype(MLLM_TYPE_F32);
        outputs[0]->alloc();
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        int start = (int)args[0];
        int end = (int)args[1];
        for (int i = 0; i < end - start; ++i) {
            outputs[0]->setDataAt<float>(0, 0, i + start, 0, (float)i);
        }
    }
};

class CPUsplitFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        int size = args.size();
        std::vector<int> each_dims;
        for (int i = 0; i < size - 2; i++) {
            each_dims.push_back(args[i]);
        }
        Chl split_dim = (Chl)args[size - 2];
        int head_size = (int)args[size - 1];
        int split_num_ = each_dims.size();
        // store each dims
        int split_dim_size_ = 0;
        std::vector<int> each_dims_;
        for (size_t i = 0; i < each_dims.size(); ++i) {
            each_dims_.push_back((float)each_dims[i]);
            split_dim_size_ += each_dims[i];
        }
        assert(split_num_ == outputs.size());
        switch (split_dim) {
        case Chl::HEAD: {
            assert(inputs[0]->head() == split_dim_size_);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), each_dims_[i], inputs[0]->sequence(), inputs[0]->dimension());
            }
            break;
        }
        case Chl::SEQUENCE: {
            assert(inputs[0]->sequence() == split_dim_size_);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), inputs[0]->head(), each_dims_[i], inputs[0]->dimension());
            }
            break;
        }
        case Chl::DIMENSION: {
            assert(inputs[0]->dimension() == split_dim_size_);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), each_dims_[i]);
            }
            break;
        }
        case Chl::D_HD: {
            assert(inputs[0]->dimension() == split_dim_size_ * head_size);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), head_size, inputs[0]->sequence(), each_dims_[i]);
            }
            break;
        }
        case Chl::HD: {
            assert(inputs[0]->dimension() == split_dim_size_ * head_size);
            for (int i = 0; i < split_num_; i++) {
                outputs[i]->reshape(inputs[0]->batch(), head_size, inputs[0]->sequence(), each_dims_[i]);
            }
            break;
        }
        default: {
            break;
        }
        }
        vector<shared_ptr<Tensor>> shared_outputs = {};
        for (const auto &output : outputs) {
            shared_outputs.push_back(std::shared_ptr<Tensor>(output, [](Tensor *) {}));
        }
        if (inputs[0]->masterTensor() == nullptr && !inputs[0]->childTensors().empty()) {
            inputs[0]->free();
        }
        inputs[0]->addTensors(shared_outputs, split_dim);
        for (const auto &output : outputs) {
            output->setDtype(MLLM_TYPE_F32);
            output->alloc();
        }
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
    }
};
} // namespace mllm
#endif // CPUTENSORFUNCTION_HPP
