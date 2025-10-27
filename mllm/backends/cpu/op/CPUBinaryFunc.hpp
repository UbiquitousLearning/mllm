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

class CPUaddFunction : public Op {
private:
    int thread_count = 4;
    float data = 0.0f; // The data to be added
public:
    CPUaddFunction(Backend *bn, string name, float data, int threadCount) :
        thread_count(threadCount), Op(bn, name) {
        this->data = data;
    }
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        auto input = inputs[0];
        auto output = outputs[0];
        output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
        output->setDtype(input->dtype());
        return ErrorCode::MLLM_NO_ERROR;
    }
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        auto input = inputs[0];
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
        for (int n = 0; n < input->batch(); ++n) {
            for (int c = 0; c < input->head(); ++c) {
                for (int h = 0; h < input->sequence(); ++h) {
                    mllm_add_fp32(input->ptrAt<float>(n, c, h, 0), data,
                                  outputs[0]->ptrAt<float>(n, c, h, 0), input->dimension());
                }
            }
        }
        return ErrorCode::MLLM_NO_ERROR;
    }
};
class CPUaddFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        float data = (float)op_param.find("data")->second;
        return new CPUaddFunction(bn, name, data, threadCount);
    }
};

class CPUsubFunction : public Op {
private:
    int thread_count = 4;
    float data = 0.0f;

public:
    CPUsubFunction(Backend *bn, string name, float data, int threadCount) :
        thread_count(threadCount), Op(bn, name) {
        this->data = data;
    }
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        auto input = inputs[0];
        auto output = outputs[0];
        output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
        output->setDtype(input->dtype());
        return ErrorCode::MLLM_NO_ERROR;
    }
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        auto input = inputs[0];
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
        for (int n = 0; n < input->batch(); ++n) {
            for (int c = 0; c < input->head(); ++c) {
                for (int h = 0; h < input->sequence(); ++h) {
                    mllm_sub_fp32(input->ptrAt<float>(n, c, h, 0), data,
                                  outputs[0]->ptrAt<float>(n, c, h, 0), input->dimension());
                }
            }
        }
        return ErrorCode::MLLM_NO_ERROR;
    }
};
class CPUsubFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        float data = (float)op_param.find("data")->second;
        return new CPUsubFunction(bn, name, data, threadCount);
    }
};

class CPUmulFunction : public Op {
private:
    int thread_count = 4;
    float data = 0.0f;

public:
    CPUmulFunction(Backend *bn, string name, float data, int threadCount) :
        thread_count(threadCount), Op(bn, name) {
        this->data = data;
    }
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        auto input = inputs[0];
        auto output = outputs[0];
        output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
        output->setDtype(input->dtype());
        return ErrorCode::MLLM_NO_ERROR;
    }
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        auto input = inputs[0];
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
        for (int n = 0; n < input->batch(); ++n) {
            for (int c = 0; c < input->head(); ++c) {
                for (int h = 0; h < input->sequence(); ++h) {
                    mllm_mul_fp32(input->ptrAt<float>(n, c, h, 0), data,
                                  outputs[0]->ptrAt<float>(n, c, h, 0), input->dimension());
                }
            }
        }
        return ErrorCode::MLLM_NO_ERROR;
    }
};
class CPUmulFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        float data = (float)op_param.find("data")->second;
        return new CPUmulFunction(bn, name, data, threadCount);
    }
};

class CPUdivFunction : public Op {
private:
    int thread_count = 4;
    float data = 0.0f;

public:
    CPUdivFunction(Backend *bn, string name, float data, int threadCount) :
        thread_count(threadCount), Op(bn, name) {
        this->data = data;
    }
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        auto input = inputs[0];
        auto output = outputs[0];
        output->setCtype(input->ctype());
        output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
        output->setDtype(input->dtype());
        return ErrorCode::MLLM_NO_ERROR;
    }
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        auto input = inputs[0];
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
        for (int n = 0; n < input->batch(); ++n) {
            for (int c = 0; c < input->head(); ++c) {
                for (int h = 0; h < input->sequence(); ++h) {
                    mllm_div_fp32(input->ptrAt<float>(n, c, h, 0), data,
                                  outputs[0]->ptrAt<float>(n, c, h, 0), input->dimension());
                }
            }
        }
        return ErrorCode::MLLM_NO_ERROR;
    }
};
class CPUdivFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        float data = (float)op_param.find("data")->second;
        return new CPUdivFunction(bn, name, data, threadCount);
    }
};

class CPUdivintFunction : public Op {
private:
    int thread_count = 4;
    float data = 0.0f;

public:
    CPUdivintFunction(Backend *bn, string name, float data, int threadCount) :
        thread_count(threadCount), Op(bn, name) {
        this->data = data;
    }
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        auto input = inputs[0];
        auto output = outputs[0];
        output->reshape(input->batch(), input->head(), input->sequence(), input->dimension());
        output->setDtype(input->dtype());
        return ErrorCode::MLLM_NO_ERROR;
    }
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        auto input = inputs[0];
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
        return ErrorCode::MLLM_NO_ERROR;
    }
};
class CPUdivintFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        float data = (float)op_param.find("data")->second;
        return new CPUdivintFunction(bn, name, data, threadCount);
    }
};

class CPUaddTwoFunction : public Op {
private:
    int thread_count = 4;

public:
    CPUaddTwoFunction(Backend *bn, string name, int threadCount) :
        thread_count(threadCount), Op(bn, name) {
    }
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        outputs[0]->reshape(std::max(inputs[0]->batch(), inputs[1]->batch()),
                            inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        return ErrorCode::MLLM_NO_ERROR;
    };
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        auto input0 = inputs[0];
        auto input1 = inputs[1];
        int batch_ = std::max(input0->batch(), input1->batch());
        int head_ = std::max(input0->head(), input1->head());
        for (int n = 0; n < batch_; ++n) {
            auto n_0 = std::min(n, input0->batch() - 1);
            auto n_1 = std::min(n, input1->batch() - 1);
#pragma omp parallel for collapse(1) num_threads(CPUBackend::cpu_threads)
            for (int c = 0; c < head_; ++c) {
                auto c_0 = std::min(c, input0->head() - 1);
                auto c_1 = std::min(c, input1->head() - 1);
                for (int h = 0; h < input0->sequence(); ++h) {
                    mllm_add_fp32(input0->ptrAt<float>(n_0, c_0, h, 0),
                                  input1->ptrAt<float>(n_1, c_1, h, 0),
                                  outputs[0]->ptrAt<float>(n, c, h, 0), input0->dimension());
                }
            }
        }
        return ErrorCode::MLLM_NO_ERROR;
    };
};
class CPUaddTwoFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new CPUaddTwoFunction(bn, name, threadCount);
    }
};

class CPUsubTwoFunction : public Op {
private:
    int thread_count = 4;

public:
    CPUsubTwoFunction(Backend *bn, string name, int threadCount) :
        thread_count(threadCount), Op(bn, name) {
    }
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        outputs[0]->reshape(std::max(inputs[0]->batch(), inputs[1]->batch()),
                            inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        return ErrorCode::MLLM_NO_ERROR;
    };
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
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
        return ErrorCode::MLLM_NO_ERROR;
    };
};
class CPUsubTwoFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new CPUsubTwoFunction(bn, name, threadCount);
    }
};

class CPUmulTwoFunction : public Op {
private:
    int thread_count = 4;

public:
    CPUmulTwoFunction(Backend *bn, string name, int threadCount) :
        thread_count(threadCount), Op(bn, name) {
    }
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        outputs[0]->reshape(std::max(inputs[0]->batch(), inputs[1]->batch()),
                            inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        return ErrorCode::MLLM_NO_ERROR;
    };
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
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
        return ErrorCode::MLLM_NO_ERROR;
    };
};
class CPUmulTwoFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new CPUmulTwoFunction(bn, name, threadCount);
    }
};

class CPUdivTwoFunction : public Op {
private:
    int thread_count = 4;

public:
    CPUdivTwoFunction(Backend *bn, string name, int threadCount) :
        thread_count(threadCount), Op(bn, name) {
    }
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        outputs[0]->reshape(std::max(inputs[0]->batch(), inputs[1]->batch()),
                            inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        return ErrorCode::MLLM_NO_ERROR;
    };
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
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
        return ErrorCode::MLLM_NO_ERROR;
    };
};
class CPUdivTwoFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new CPUdivTwoFunction(bn, name, threadCount);
    }
};

} // namespace mllm
#endif // CPUBINARYFUNC_HPP