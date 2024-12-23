//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUBINARYFUNC_HPP
#define CPUBINARYFUNC_HPP
#include "CPUBackend.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "../compute/Arithmetic.hpp"

namespace mllm {
class Tensor;

class CPUaddFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        // float data = (float)args[0];
        auto input = inputs[0];
        auto output = outputs[0];
        output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
        output->setDtype(input->dtype());
        output->alloc();
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
    }
};
class CPUsubFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        auto input = inputs[0];
        auto output = outputs[0];
        output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
        output->setDtype(input->dtype());
        output->alloc();
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
    }
};
class CPUmulFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        auto input = inputs[0];
        auto output = outputs[0];
        output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
        output->setDtype(input->dtype());
        output->alloc();
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
    }
};
class CPUdivFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        auto input = inputs[0];
        auto output = outputs[0];
        output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
        output->setDtype(input->dtype());
        output->alloc();
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
    }
};

class CPUdivintFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        auto input = inputs[0];
        auto output = outputs[0];
        output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
        output->setDtype(input->dtype());
        output->alloc();
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        float data = (float)args[0];
        auto input = inputs[0];
        auto output = outputs[0];
#pragma omp parallel for collapse(4) num_threads(CPUBackend::cpu_threads)
        for (int n = 0; n < input->batch(); ++n) {
            for (int c = 0; c < input->head(); ++c) {
                for (int h = 0; h < input->sequence(); ++h) {
                    for (int d = 0; d < input->dimension(); ++d) {
                        outputs[0]->setDataAt<float>(n, c, h, d,
                                                     static_cast<int>(input->dataAt<float>(n, c, h, d) / data));
                    }
                }
            }
        }
    }
};

class CPUaddTwoFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        outputs[0]->reshape(std::max(inputs[0]->batch(), inputs[1]->batch()),
                            inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
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
                    mllm_add_fp32(input0->ptrAt<float>(n_0, c, h, 0),
                                  input1->ptrAt<float>(n_1, c, h, 0),
                                  outputs[0]->ptrAt<float>(n, c, h, 0), input0->dimension());
                }
            }
        }
    };
};
class CPUsubTwoFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        outputs[0]->reshape(std::max(inputs[0]->batch(), inputs[1]->batch()),
                            inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
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
                    mllm_sub_fp32(input0->ptrAt<float>(n_0, c, h, 0),
                                  input1->ptrAt<float>(n_1, c, h, 0),
                                  outputs[0]->ptrAt<float>(n, c, h, 0), input0->dimension());
                }
            }
        }
    };
};
class CPUmulTwoFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        outputs[0]->reshape(std::max(inputs[0]->batch(), inputs[1]->batch()),
                            inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
    };
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        if (outputs[0]->sequence() == 0
            || inputs[0]->sequence() != outputs[0]->sequence()) {
            outputs[0]->reshape(std::max(inputs[0]->batch(), inputs[1]->batch()),
                                inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
            outputs[0]->alloc();
        }
        auto input0 = inputs[0];
        auto input1 = inputs[1];
        int batch_ = std::max(input0->batch(), input1->batch());
        for (int n = 0; n < batch_; ++n) {
            auto n_0 = std::min(n, input0->batch() - 1);
            auto n_1 = std::min(n, input1->batch() - 1);
#pragma omp parallel for collapse(2) num_threads(CPUBackend::cpu_threads)
            for (int c = 0; c < input0->head(); ++c) {
                for (int h = 0; h < input0->sequence(); ++h) {
                    if (input1->dimension() == 1) {
                        mllm_mul_fp32(input0->ptrAt<float>(n_0, c, h, 0),
                                      input1->dataAt<float>(n_1, c, h, 0),
                                      outputs[0]->ptrAt<float>(n, c, h, 0), input0->dimension());
                    } else {
                        mllm_mul_fp32(input0->ptrAt<float>(n_0, c, h, 0),
                                      input1->ptrAt<float>(n_1, c, h, 0),
                                      outputs[0]->ptrAt<float>(n, c, h, 0), input0->dimension());
                    }
                }
            }
        }
    };
};
class CPUdivTwoFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        outputs[0]->reshape(std::max(inputs[0]->batch(), inputs[1]->batch()),
                            inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
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
                    if (input1->dimension() == 1) {
                        mllm_div_fp32(input0->ptrAt<float>(n_0, c, h, 0),
                                      input1->dataAt<float>(n_1, c, h, 0),
                                      outputs[0]->ptrAt<float>(n, c, h, 0), input0->dimension());
                    } else {
                        mllm_div_fp32(input0->ptrAt<float>(n_0, c, h, 0),
                                      input1->ptrAt<float>(n_1, c, h, 0),
                                      outputs[0]->ptrAt<float>(n, c, h, 0), input0->dimension());
                    }
                }
            }
        }
    };
};

} // namespace mllm
#endif // CPUBINARYFUNC_HPP