//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUTENSORFUNCTION_HPP
#define CPUTENSORFUNCTION_HPP
#include "CPUBackend.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "compute/Matmul.hpp"

// #include <Layer.hpp>
#include <iostream>
#include <vector>

namespace mllm {
class Tensor;

class CPUmmFunction: public TensorFunction {
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
    void setup(Tensor &output, Tensor &input0, Tensor &input1) {
        if (input1.chls()[SEQUENCE] != 3) {
            tranTensorChl(input1);
        }
        assert(input0.dimension() == input1.sequence());
        output.reshape(input0.batch(), input0.head(), input0.sequence(), input1.dimension());
        output.setDtype(input0.dtype());
        output.alloc();
    }
    void execute(Tensor &output, Tensor &input0, Tensor &input1) {
        bool isSame = std::equal(input0.chls().begin(), input0.chls().end(), input1.chls().begin());
        assert(input0.dtype() == MLLM_TYPE_F32);
        switch (input1.dtype()) {
        case MLLM_TYPE_F32: {
            mat_mul_fp32(&input0, &input1, &output, false, nullptr, false, isSame, CPUBackend::cpu_threads);
            break;
        }
        case MLLM_TYPE_F16: {
            mat_mul_fp32_fp16(&input0, &input1, &output, false, nullptr, false, isSame, CPUBackend::cpu_threads);
            break;
        }
        default:
            break;
        }
    }
    void setup(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        setup(output, *inputs[0], *inputs[1]);
    }
    void execute(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        execute(output, *inputs[0], *inputs[1]);
    }
    
};

class CPUnormFunction: public TensorFunction {
public:
    void setup(Tensor &input, Tensor &output, int L_n) {
        output.reshape(input.batch(), input.head(), input.sequence(), input.dimension());
        output.setDtype(input.dtype());
        output.alloc();
    }
    void execute(Tensor &input, Tensor &output, int L_n) {
        for (int h = 0; h < input.head(); h++) {
            for (int n = 0; n < input.batch(); n++) {
                for (int s = 0; s < input.sequence(); s++) {
                    if (L_n == 2) {
                        float sum_of_squares = 0.0f;
                        for (int d = 0; d < input.dimension(); ++d) {
                            sum_of_squares += input.dataAt<float>(n, h, s, d) * input.dataAt<float>(n, h, s, d);
                        }
                        float l2_norm = std::sqrt(sum_of_squares);
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
                        for (int d = 0; d < input.dimension(); d++) {
                            output.setDataAt<float>(n, h, s, d, l2_norm);
                        }
                    } else {
                        float sum_of_abs_values = 0.0f;
                        for (int d = 0; d < input.dimension(); ++d) {
                            sum_of_abs_values += std::abs(input.dataAt<float>(n, h, s, d));
                        }
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
                        for (int d = 0; d < input.dimension(); d++) {
                            output.setDataAt<float>(n, h, s, d, sum_of_abs_values);
                        }
                    }
                }
            }
        }
    }
    void setup(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        setup( *inputs[0], output,(int)args[0]);
    }
    void execute(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        execute(*inputs[0], output,(int)args[0]);
    }
};

class CPUbinaryFunction {
public:
    template <typename Func>
    void setup(Tensor &input, Tensor &output, Func operation, float data) {
        output.reshape(input.batch(), input.head(), input.sequence(), input.dimension());
        output.setDtype(input.dtype());
        output.alloc();
    }
    template <typename Func>
    void execute(Tensor &input, Tensor &output, Func operation, float data) {
        if (input.masterTensor() == nullptr && output.masterTensor() == nullptr && input.ctype() == output.ctype()) {
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
            for (int is = 0; is < input.batch() * input.head() * input.sequence() * input.dimension(); ++is) {
                output.hostPtr<float>()[is] = operation(input.hostPtr<float>()[is], data);
            }
        } else {
            for (int n = 0; n < input.batch(); ++n) {
                for (int c = 0; c < input.head(); ++c) {
                    for (int h = 0; h < input.sequence(); ++h) {
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
                        for (int w = 0; w < input.dimension(); ++w) {
                            output.ptrAt<float>(n, c, h, w)[0] =
                                operation(input.ptrAt<float>(n, c, h, w)[0],
                                          data);
                        }
                    }
                }
            }
        }
    }
};

class CPUaddFunction: public TensorFunction, public CPUbinaryFunction {
public:
    void setup(Tensor &input, Tensor &output, float data) {
        CPUbinaryFunction::setup( input, output, std::plus<float>(), data);
    };
    void execute(Tensor &input, Tensor &output,float data) {
        CPUbinaryFunction::execute( input, output, std::plus<float>(), data);
    };
    void setup(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        setup( *inputs[0], output,(float)args[0]);
    }
    void execute(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        execute(*inputs[0], output,(float)args[0]);
    }
};
class CPUsubFunction: public TensorFunction, public CPUbinaryFunction {
public:
    void setup(Tensor &input, Tensor &output, float data) {
        CPUbinaryFunction::setup( input, output, std::minus<float>(), data);
    };
    void execute(Tensor &input, Tensor &output,float data) {
        CPUbinaryFunction::execute( input, output, std::minus<float>(), data);
    };
    void setup(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        setup( *inputs[0], output,(float)args[0]);
    }
    void execute(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        execute(*inputs[0], output,(float)args[0]);
    }
};
class CPUmulFunction: public TensorFunction, public CPUbinaryFunction {
public:
    void setup(Tensor &input, Tensor &output, float data) {
        CPUbinaryFunction::setup( input, output, std::multiplies<float>(), data);
    };
    void execute(Tensor &input, Tensor &output,float data) {
        CPUbinaryFunction::execute( input, output, std::multiplies<float>(), data);
    };
    void setup(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        setup( *inputs[0], output,(float)args[0]);
    }
    void execute(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        execute(*inputs[0], output,(float)args[0]);
    }
};
class CPUdivFunction: public TensorFunction, public CPUbinaryFunction {
public:
    void setup(Tensor &input, Tensor &output, float data) {
        CPUbinaryFunction::setup( input, output, std::divides<float>(), data);
    };
    void execute(Tensor &input, Tensor &output,float data) {
        CPUbinaryFunction::execute( input, output, std::divides<float>(), data);
    };
    void setup(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        setup( *inputs[0], output,(float)args[0]);
    }
    void execute(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        execute(*inputs[0], output,(float)args[0]);
    }
};

class CPUbinaryTwoFunction {
public:
    template <typename Func>
    void setup(Tensor &input0, Tensor &output, Tensor &input1, Func operation) {
        output.reshape(std::max(input0.batch(), input1.batch()), input0.head(), input0.sequence(), input0.dimension());
        output.setDtype(input0.dtype());
        output.alloc();
    }
    template <typename Func>
    void execute(Tensor &input0, Tensor &output, Tensor &input1, Func operation) {
        int batch_ = std::max(input0.batch(), input1.batch());
        if (input0.masterTensor() == nullptr && output.masterTensor() == nullptr && input0.ctype() == output.ctype()) {
            for (int n = 0; n < batch_; ++n) {
                auto n_0 = std::min(n, input0.batch() - 1);
                auto n_1 = std::min(n, input1.batch() - 1);
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
                for (int is = 0; is < input0.head() * input0.sequence() * input0.dimension(); ++is) {
                    output.ptrAt<float>(n, 0, 0, 0)[is] =
                        operation(input0.ptrAt<float>(n_0, 0, 0, 0)[is],
                                  input1.ptrAt<float>(n_1, 0, 0, 0)[is]);
                }
            }
        } else {
            for (int n = 0; n < batch_; ++n) {
                auto n_0 = std::min(n, input0.batch() - 1);
                auto n_1 = std::min(n, input1.batch() - 1);
                for (int c = 0; c < input0.head(); ++c) {
                    for (int h = 0; h < input0.sequence(); ++h) {
#pragma omp parallel for num_threads(CPUBackend::cpu_threads)
                        for (int w = 0; w < input0.dimension(); ++w) {
                            output.ptrAt<float>(n, c, h, w)[0] =
                                operation(input0.ptrAt<float>(n_0, c, h, w)[0],
                                          input1.ptrAt<float>(n_1, c, h, w)[0]);
                        }
                    }
                }
            }
        }
    }
};
class CPUaddTwoFunction: public TensorFunction, public CPUbinaryTwoFunction {
public:
    void setup(Tensor &input0, Tensor &output, Tensor &input1) {
        CPUbinaryTwoFunction::setup( input0, output, input1, std::plus<float>());
    };
    void execute(Tensor &input0, Tensor &output, Tensor &input1) {
        CPUbinaryTwoFunction::execute( input0, output, input1, std::plus<float>());
    };
    void setup(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        setup( *inputs[0], output, *inputs[1]);
    }
    void execute(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        execute(*inputs[0], output, *inputs[1]);
    }
};
class CPUsubTwoFunction: public TensorFunction, public CPUbinaryTwoFunction {
public:
    void setup(Tensor &input0, Tensor &output, Tensor &input1) {
        CPUbinaryTwoFunction::setup( input0, output, input1, std::minus<float>());
    };
    void execute(Tensor &input0, Tensor &output, Tensor &input1) {
        CPUbinaryTwoFunction::execute( input0, output, input1, std::minus<float>());
    };
    void setup(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        setup( *inputs[0], output, *inputs[1]);
    }
    void execute(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        execute(*inputs[0], output, *inputs[1]);
    }
};
class CPUmulTwoFunction: public TensorFunction, public CPUbinaryTwoFunction {
public:
    void setup(Tensor &input0, Tensor &output, Tensor &input1) {
        CPUbinaryTwoFunction::setup( input0, output, input1, std::multiplies<float>());
    };
    void execute(Tensor &input0, Tensor &output, Tensor &input1) {
        CPUbinaryTwoFunction::execute( input0, output, input1, std::multiplies<float>());
    };
    void setup(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        setup( *inputs[0], output, *inputs[1]);
    }
    void execute(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        execute(*inputs[0], output, *inputs[1]);
    }
};
class CPUdivTwoFunction: public TensorFunction, public CPUbinaryTwoFunction {
public:
    void setup(Tensor &input0, Tensor &output, Tensor &input1) {
        CPUbinaryTwoFunction::setup( input0, output, input1, std::divides<float>());
    };
    void execute(Tensor &input0, Tensor &output, Tensor &input1) {
        CPUbinaryTwoFunction::execute( input0, output, input1, std::divides<float>());
    };
    void setup(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        setup( *inputs[0], output, *inputs[1]);
    }
    void execute(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        execute(*inputs[0], output, *inputs[1]);
    }
};

class CPUmeanFunction: public TensorFunction {
public:
    void setup(Tensor &input, Tensor &output, Chl axis) {
        int batch = input.batch();
        int head = input.head();
        int sequence = input.sequence();
        int dimension = input.dimension();
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
        output.reshape(batch, head, sequence, dimension);
        output.setDtype(input.dtype());
        output.alloc();
    }
    void execute(Tensor &input, Tensor &output, Chl axis) {
        int batch = input.batch();
        int dim = input.dimension();
        int seq = input.sequence();
        int head = input.head();
        switch (axis) {
        case BATCH: {
            for (int h = 0; h < head; h++) {
                for (int s = 0; s < seq; ++s) {
                    for (int d = 0; d < dim; d++) {
                        float sum = 0.0f;
                        for (int n = 0; n < batch; n++) {
                            sum += input.dataAt<float>(n, h, s, d);
                        }
                        output.setDataAt<float>(0, h, s, d, sum / seq);
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
                            sum += input.dataAt<float>(n, h, s, d);
                        }
                        output.setDataAt<float>(n, 0, s, d, sum / seq);
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
                            sum += input.dataAt<float>(n, h, s, d);
                        }
                        output.setDataAt<float>(n, h, 0, d, sum / seq);
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
                        for (int d = 0; d < input.dimension(); ++d) {
                            sum += input.dataAt<float>(n, h, s, d);
                        }
                        output.setDataAt<float>(n, h, s, 0, sum / input.dimension());
                    }
                }
            }
            break;
        }
        default:
            break;
        }
    }
    void setup(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        setup( *inputs[0], output, (Chl)args[0]);
    }
    void execute(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        execute(*inputs[0], output, (Chl)args[0]);
    }
};

class CPUviewFunction: public TensorFunction{
public:
    void setup(Tensor &input, Tensor &output, int b, int h, int s, int d) {
        int dim_b = input.batch();
        int dim_h = input.head();
        int dim_s = input.sequence();
        int dim_d = input.dimension();
        if (b == -1 && h != -1 && s == -1 && d != -1) { // head & dimension
            if (h != ANYDIM && d != ANYDIM) {
                assert(input.dimension() * input.head() == h * d);
                dim_h = h;
                dim_d = d;
            } else if (h != ANYDIM) {
                dim_h = h;
                dim_d = input.dimension() * input.head() / h;
            } else if (d != ANYDIM) {
                dim_h = input.dimension() * input.head() / d;
                dim_d = d;
            } else {
                std::cout << "[TODO]Tensor.View not support!!!!" << std::endl;
            }
        } else if (b == -1 && h != -1 && s != -1 && d == -1) { // head & sequence
            if (h != ANYDIM && s != ANYDIM) {
                assert(input.sequence() * input.head() == h * s);
                dim_h = h;
                dim_s = s;
            } else if (h != ANYDIM) {
                dim_h = h;
                dim_s = input.sequence() * input.head() / h;
            } else if (s != ANYDIM) {
                dim_h = input.sequence() * input.head() / s;
                dim_s = s;
            } else {
                std::cout << "[TODO]Tensor.View not support!!!!" << std::endl;
            }
        } else if (b != -1 && h == -1 && s != -1 && d == -1) { // batch & sequence
            if (b != ANYDIM && s != ANYDIM) {
                assert(input.sequence() * input.batch() == b * s);
                dim_b = b;
                dim_s = s;
            } else if (b != ANYDIM) {
                dim_b = b;
                dim_s = input.sequence() * input.batch() / b;
            } else if (s != ANYDIM) {
                dim_b = input.sequence() * input.batch() / s;
                dim_s = s;
            } else {
                std::cout << "[TODO]Tensor.View not support!!!!" << std::endl;
            }
        } else {
            std::cout << "[TODO]Tensor.View not support!!!!" << std::endl;
        }
        output.reshape(dim_b, dim_h, dim_s, dim_d);
        if ((b == -1 && s == -1 && input.ctype() != BCTHW)   // head & dimension
            || (b == -1 && d == -1 && input.ctype() == BSHD) // head & sequence
            || (h == -1 && d == -1 && input.ctype() == BSHD) // batch & sequence
        ) {
            if (input.masterTensor() == nullptr) {
                input.free();
            }
            output.setDtype(input.dtype());
            output.alloc();
            input.deepCopyFrom(output, false);
        } else {
            std::cout << "[TODO]Tensor.View not support!!!!" << std::endl;
        }
    }
    void execute(Tensor &input, Tensor &output, int b, int h, int s, int d) {
    }
    void setup(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        setup( *inputs[0], output, (int)args[0], (int)args[1], (int)args[2], (int)args[3]);
    }
    void execute(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        execute(*inputs[0], output, (int)args[0], (int)args[1], (int)args[2], (int)args[3]);
    }
};

class CPUflattenFunction: public TensorFunction{
public:
    void setup(Tensor &input, Tensor &output, Chl axis_start, Chl axis_end) {
        int dim_b = input.batch();
        int dim_h = 0;
        int dim_s = 0;
        int dim_d = 0;
        if (input.shape().size() == 4) {
            dim_h = input.head();
            dim_s = input.sequence();
            dim_d = input.dimension();
            if (axis_start == BATCH & axis_end == SEQUENCE) {
                dim_b = 1;
                dim_s = input.sequence() * input.batch();
            } else if (axis_start == HEAD & axis_end == SEQUENCE) {
                dim_h = 1;
                dim_s = input.sequence() * input.head();
            } else if (axis_start == HEAD & axis_end == DIMENSION) {
                dim_h = 1;
                dim_d = input.dimension() * input.head();
            } else {
                std::cout << "ERROR:  flatten  " << axis_start << "&" << axis_end << std::endl;
            }
        } else if (input.shape().size() == 5) {
            if (axis_start == CHANNLE & axis_end == HEIGHT) {
                dim_h = 1;
                dim_s = input.channel() * input.height() * input.time();
                dim_d = input.width();
            } else if (axis_start == HEIGHT & axis_end == CHANNLE) {
                dim_h = 1;
                dim_s = input.channel() * input.height() * input.width();
                dim_d = input.time();
            }
        }
        assert(dim_d + dim_s + dim_h > 0);
        output.reshape(dim_b, dim_h, dim_s, dim_d);
        if ((axis_start == TIME & axis_end == WIDTH && input.ctype() == BCTHW)
            || (axis_start == CHANNLE & axis_end == HEIGHT && input.ctype() == BWCTH)
            || (axis_start == HEIGHT & axis_end == CHANNLE && input.ctype() == BTHWC)
            || (axis_start == BATCH & axis_end == SEQUENCE && input.ctype() != BCTHW)
            || (axis_start == HEAD & axis_end == SEQUENCE && input.ctype() == BSHD)
            || (axis_start == HEAD & axis_end == SEQUENCE && input.ctype() == BHDS)
            || (axis_start == HEAD & axis_end == SEQUENCE && input.ctype() == BDHS)
            || (axis_start == HEAD & axis_end == DIMENSION && input.ctype() == BSHD)
            || (axis_start == HEAD & axis_end == DIMENSION && input.ctype() == BHDS)
            || (axis_start == HEAD & axis_end == SEQUENCE && input.ctype() == BDSH)) {
            if (input.masterTensor() == nullptr) {
                input.free();
            }
            output.setDtype(input.dtype());
            output.alloc();
            input.deepCopyFrom(output, false);
        } else {
            std::cout << "[TODO]Tensor.Flatten not support!!!!" << std::endl;
        }
    }
    void execute(Tensor &input, Tensor &output, Chl axis_start, Chl axis_end) {
    }
    void setup(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        setup( *inputs[0], output, (Chl)args[0], (Chl)args[1]);
    }
    void execute(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        execute(*inputs[0], output,(Chl)args[0], (Chl)args[1]);
    }
};
class CPUtransposeFunction : public TensorFunction {
public:
    void setup(Tensor &input, Tensor &output, vector<std::pair<Chl, Chl>> axiss) {
        if (output.count() <= 0 || output.shape() != input.shape()) {
            output.trans_copy_shape(input.shape());
            std::map<Chl, int> origin_chls = {{BATCH, 0}, {SEQUENCE, 1}, {HEAD, 2}, {DIMENSION, 3}, {CHANNLE, 1}, {TIME, 2}, {HEIGHT, 3}, {WIDTH, 4}};
            if (std::equal(output.chls().begin(), output.chls().end(), origin_chls.begin())) {
                output.chls() = input.chls();
                for (auto axis : axiss) {
                    auto axis0 = axis.first;
                    auto axis1 = axis.second;
                    auto ori_0_idx = output.chls()[axis0];
                    auto ori_1_idx = output.chls()[axis1];
                    output.chls()[axis0] = ori_1_idx;
                    output.chls()[axis1] = ori_0_idx;
                }
                output.changeCtype(input.shape().size());
                output.undiffusion() = true;
            }
            if (input.masterTensor() != nullptr) {
                if (output.masterTensor() == nullptr) {
                    output.setDtype(input.dtype());
                    output.deepCopyFrom(input, false);
                }
            } else {
                if (input.masterTensor() == nullptr) {
                    input.free();
                }
                output.setDtype(input.dtype());
                output.alloc();
                // input.undiffusion() = true;
                input.setUndiffusion(true);
                input.deepCopyFrom(output, false);
                output.transFrom() = axiss;
            }
        }
    }
    void execute(Tensor &input, Tensor &output, vector<std::pair<Chl, Chl>> axiss) {
    }
    void setup(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        vector<std::pair<Chl, Chl>> axiss;
        for (int i = 0; i < args.size(); i += 2) {
            axiss.push_back({(Chl)args[i], (Chl)args[i + 1]});
        }
        setup( *inputs[0], output, axiss);
    }
    void execute(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        vector<std::pair<Chl, Chl>> axiss;
        for (int i = 0; i < args.size(); i += 2) {
            axiss.push_back({(Chl)args[i], (Chl)args[i + 1]});
        }
        execute( *inputs[0], output, axiss);
    }
    
};

class CPUclipFunction : public TensorFunction {
public:
    void setup(Tensor &input, Tensor &output, vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
        // reshape
        int dim_b = input.batch();
        int dim_h = input.head();
        int dim_s = input.sequence();
        int dim_d = input.dimension();
        std::vector<std::pair<std::vector<int>, int *>> data = {{b, &dim_b}, {h, &dim_h}, {s, &dim_s}, {d, &dim_d}};
        for (auto &pair : data) {
            if (pair.first.size() == 2) {
                *pair.second = pair.first[1] - pair.first[0];
            } else if (pair.first.size() == 1) {
                *pair.second = 1;
            }
        }
        output.reshape(dim_b, dim_h, dim_s, dim_d);
        output.setDtype(input.dtype());
        output.alloc();
    }
    void execute(Tensor &input, Tensor &output, vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
        if (s.size() == 2) {
            for (int b = 0; b < input.batch(); ++b) {
                memcpy(output.hostPtr<float>() + output.offset(b, 0, 0, 0),
                       input.hostPtr<float>() + input.offset(b, 0, s[0], 0),
                       input.head() * (s[1] - s[0]) * input.dimension() * sizeof(float));
            }
        } else if (s.size() == 1) {
            int seq_idx = s[0];
            if (seq_idx < 0) {
                seq_idx = input.sequence() + seq_idx;
            }
            for (int b = 0; b < input.batch(); ++b) {
                memcpy(output.hostPtr<float>() + output.offset(b, 0, 0, 0),
                       input.hostPtr<float>() + input.offset(b, 0, seq_idx, 0),
                       input.head() * 1 * input.dimension() * sizeof(float));
            }
        } else {
            std::cout << "[TODO]Tensor.CLip not support!!!!" << std::endl;
        }
    }
    void setup(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        int b_size = args[0];
        int h_size = args[1];
        int s_size = args[2];
        int d_size = args[3];
        vector<int> b = {};
        vector<int> h = {};
        vector<int> s = {};
        vector<int> d = {};
        for (int i=0; i<b_size; i++) {
            b.push_back(args[4+i]);        
        }
        for (int i=0; i<h_size; i++) {
            h.push_back(args[4+b_size+i]);        
        }
        for (int i=0; i<s_size; i++) {
            s.push_back(args[4+b_size+h_size+i]);        
        }
        for (int i=0; i<d_size; i++) {
            d.push_back(args[4+b_size+h_size+s_size+i]);        
        }
        setup( *inputs[0], output, b, h, s, d);
    }
    void execute(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        int b_size = args[0];
        int h_size = args[1];
        int s_size = args[2];
        int d_size = args[3];
        vector<int> b = {};
        vector<int> h = {};
        vector<int> s = {};
        vector<int> d = {};
        for (int i=0; i<b_size; i++) {
            b.push_back(args[4+i]);        
        }
        for (int i=0; i<h_size; i++) {
            h.push_back(args[4+b_size+i]);        
        }
        for (int i=0; i<s_size; i++) {
            s.push_back(args[4+b_size+h_size+i]);        
        }
        for (int i=0; i<d_size; i++) {
            d.push_back(args[4+b_size+h_size+s_size+i]);        
        }
        execute( *inputs[0], output, b, h, s, d);
    }
};

class CPUclipaxisFunction: public TensorFunction{
public:
    void setup(Tensor &input, Tensor &output, Chl axis, vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
        // reshape
        int dim_b = input.batch();
        int dim_h = input.head();
        int dim_s = input.sequence();
        int dim_d = input.dimension();
        switch (axis) {
        case BATCH: {
            std::vector<std::pair<std::vector<int>, int *>> data = {{h, &dim_h}, {s, &dim_s}, {d, &dim_d}};
            for (auto &pair : data) {
                if (pair.first.size() > 0) {
                    *pair.second = 1;
                }
            }
            break;
        }
        case HEAD: {
            std::vector<std::pair<std::vector<int>, int *>> data = {{b, &dim_b}, {s, &dim_s}, {d, &dim_d}};
            for (auto &pair : data) {
                if (pair.first.size() > 0) {
                    *pair.second = 1;
                }
            }
            break;
        }
        case SEQUENCE: {
            std::vector<std::pair<std::vector<int>, int *>> data = {{b, &dim_b}, {h, &dim_h}, {d, &dim_d}};
            for (auto &pair : data) {
                if (pair.first.size() > 0) {
                    *pair.second = 1;
                }
            }
            break;
        }
        case DIMENSION: {
            std::vector<std::pair<std::vector<int>, int *>> data = {{b, &dim_b}, {h, &dim_h}, {s, &dim_s}};
            for (auto &pair : data) {
                if (pair.first.size() > 0) {
                    *pair.second = 1;
                }
            }
            break;
        }
        default:
            break;
        }
        output.reshape(dim_b, dim_h, dim_s, dim_d);
        output.setDtype(input.dtype());
        output.alloc();
    }
    void execute(Tensor &input, Tensor &output, Chl axis, vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
        if (axis == BATCH) {
            if (s.size() > 0) {
                for (int i = 0; i < s.size(); ++i) {
                    auto seq_idx = s[i];
                    memcpy(output.hostPtr<float>() + output.offset(i, 0, 0, 0),
                           input.hostPtr<float>() + input.offset(i, 0, seq_idx, 0),
                           input.head() * 1 * input.dimension() * sizeof(float));
                }
            }
        } else {
            std::cout << "[TODO]Tensor.CLip not support!!!!" << std::endl;
        }
    }
    void setup(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {
        Chl axis = (Chl)args[0];
        int b_size = args[1];
        int h_size = args[2];
        int s_size = args[3];
        int d_size = args[4];
        vector<int> b = {};
        vector<int> h = {};
        vector<int> s = {};
        vector<int> d = {};
        for (int i=0; i<b_size; i++) {
            b.push_back(args[5+i]);        
        }
        for (int i=0; i<h_size; i++) {
            h.push_back(args[5+b_size+i]);        
        }
        for (int i=0; i<s_size; i++) {
            s.push_back(args[5+b_size+h_size+i]);        
        }
        for (int i=0; i<d_size; i++) {
            d.push_back(args[5+b_size+h_size+s_size+i]);        
        }
        setup( *inputs[0], output, axis, b, h, s, d);    
    }
    void execute(Tensor &output, vector<Tensor*> &inputs, vector<float> args) override {        
        Chl axis = (Chl)args[0];
        int b_size = args[1];
        int h_size = args[2];
        int s_size = args[3];
        int d_size = args[4];
        vector<int> b = {};
        vector<int> h = {};
        vector<int> s = {};
        vector<int> d = {};
        for (int i=0; i<b_size; i++) {
            b.push_back(args[5+i]);        
        }
        for (int i=0; i<h_size; i++) {
            h.push_back(args[5+b_size+i]);        
        }
        for (int i=0; i<s_size; i++) {
            s.push_back(args[5+b_size+h_size+i]);        
        }
        for (int i=0; i<d_size; i++) {
            d.push_back(args[5+b_size+h_size+s_size+i]);        
        }
        execute( *inputs[0], output, axis, b, h, s, d);        
    }
};

class CPUcatFunction: public TensorFunction {
public:
    void setup(Tensor &output, vector<Tensor *> inputs, Chl axis) {
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
        output.reshape(dim_b, dim_h, dim_s, dim_d);
        output.setDtype(inputs[0]->dtype());
        output.alloc();
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
                inputs[idx]->deepCopyFrom(output, false, {cbatch, chead, cseq, cdim}); // b,h,s,d
            }
        }
        else if (axis == DIMENSION && inputs[0]->head() != 1) {
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
                if (inputs[idx]->deaggregated_tensor() != nullptr) {
                    for (int t=0; t<inputs[idx]->deaggregated_tensor()->aggregated_tensors().size(); t++ ) {
                        if(inputs[idx]->deaggregated_tensor()->aggregated_tensors()[t].get()==inputs[idx]){
                            tmp_agg_idx = t;
                            continue;
                        }
                    }
                }
                inputs[idx]->deepCopyFrom(output, false, {cbatch, chead, cseq, cdim}); // b,h,s,d
                if (inputs[idx]->deaggregated_tensor() != nullptr) {
                    vector<shared_ptr<Tensor>> shared_outputs = {};
                    for (int t=0; t<inputs[idx]->deaggregated_tensor()->aggregated_tensors().size(); t++ ) {
                        if(t==tmp_agg_idx){
                            inputs[idx]->deaggregated_tensor()->aggregated_tensors()[t] = 
                                std::shared_ptr<Tensor>(inputs[idx], [](Tensor *) {});
                        }
                    }
                }
            }
        }
    }
    void execute(Tensor &output, vector<Tensor *> inputs, Chl axis) {
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
                memcpy(output.ptrAt<float>(n * inputs[0]->batch(), 0, 0, 0),
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
                            memcpy(output.ptrAt<float>(n, c, h, w),
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
                    memcpy(output.ptrAt<float>(n, 0, h, 0),
                           inputs[idx]->ptrAt<float>(n_, 0, 0, 0),
                           sizeof(float) * (inputs[idx]->sequence() * inputs[idx]->dimension()));
                    h += inputs[idx]->sequence();
                }
            }
        }
    }
    void setup(Tensor &output, vector<Tensor *> &inputs, vector<float> args) override {
        setup( output, inputs, (Chl)args[0]);
    }
    void execute(Tensor &output, vector<Tensor *> &inputs, vector<float> args) override {
        execute( output, inputs, (Chl)args[0]);
    }
};

class CPUwhereFunction: public TensorFunction {
public:
    void setup(Tensor &input, Tensor &output, float value, Chl axis) {
    }
    void execute(Tensor &input, Tensor &output, float value, Chl axis) {
        vector<float> b_vec = {};
        vector<float> s_vec = {};
        vector<float> h_vec = {};
        vector<float> d_vec = {};
#pragma omp parallel for collapse(4) num_threads(CPUBackend::cpu_threads)
        for (int b = 0; b < input.batch(); b++) {
            for (auto s = 0; s < input.sequence(); s++) {
                for (auto h = 0; h < input.head(); h++) {
                    for (auto d = 0; d < input.dimension(); d++) {
                        if (input.dataAt<float>(b, h, h, s) == value) {
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
            output.reshape(1, 1, 4, num);
            output.setDtype(input.dtype());
            output.alloc();
            for (int i = 0; i < 4; ++i) {
                auto dest_ptr = output.hostPtr<float>() + output.offset(0, 0, i, 0);
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
            output.reshape(1, 1, 1, num);
            output.setDtype(input.dtype());
            output.alloc();
            auto dest_ptr = output.hostPtr<float>();
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
    void setup(Tensor &output, vector<Tensor *> &inputs, vector<float> args) override {
        setup( *inputs[0], output, args[0], (Chl)args[1]);
    }
    void execute(Tensor &output, vector<Tensor *> &inputs, vector<float> args) override {
        execute( *inputs[0], output, args[0], (Chl)args[1]);
    }
};

class CPURangeFunction: public TensorFunction {
public:
    void setup(Tensor &output, int start, int end) {
        output.reshape(1, 1, end - start, 1);
        output.setDtype(MLLM_TYPE_F32);
        output.alloc();
    }
    void execute(Tensor &output, int start, int end) {
        for (int i = 0; i < end - start; ++i) {
            output.setDataAt<float>(0, 0, i + start, 0, (float)i);
        }
    }
    void setup(Tensor &output, vector<Tensor *> &inputs, vector<float> args) override {
        setup( output, args[0], args[1]);
    }
    void execute(Tensor &output, vector<Tensor *> &inputs, vector<float> args) override {
        execute( output, args[0], args[1]);
    }
};

class CPUsplitFunction: public TensorFunction {
public:
    void setup(Tensor &input, vector<Tensor *> &outputs, const std::vector<int> &each_dims, Chl split_dim, int head_size) {
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
            assert(input.head() == split_dim_size_);
            for (int i=0; i<split_num_; i++) {
                outputs[i]->reshape(input.batch(), each_dims_[i], input.sequence(), input.dimension());
            }
            break;
        }
        case Chl::SEQUENCE: {
            assert(input.sequence() == split_dim_size_);
            for (int i=0; i<split_num_; i++) {
                outputs[i]->reshape(input.batch(), input.head(), each_dims_[i], input.dimension());
            }
            break;
        }
        case Chl::DIMENSION: {
            assert(input.dimension() == split_dim_size_);
            for (int i=0; i<split_num_; i++) {
                outputs[i]->reshape(input.batch(), input.head(), input.sequence(), each_dims_[i]);
            }
            break;
        }
        case Chl::D_HD: {
            assert(input.dimension() == split_dim_size_*head_size);
            for (int i=0; i<split_num_; i++) {
                outputs[i]->reshape(input.batch(), head_size, input.sequence(), each_dims_[i]);
            }
            break;
        }
        case Chl::HD: {
            assert(input.dimension() == split_dim_size_*head_size);
            for (int i=0; i<split_num_; i++) {
                outputs[i]->reshape(input.batch(), head_size, input.sequence(), each_dims_[i]);
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
        if (input.masterTensor() == nullptr && input.childTensors().size() > 0) {
            input.free();
        }
        input.addTensors(shared_outputs, split_dim);
        for (const auto &output : outputs) {
            output->setDtype(MLLM_TYPE_F32);
            output->alloc();
        }
    }
    void setup(vector<Tensor*> &output, vector<Tensor*> &inputs, vector<float> args) override{
        int size = args.size();
        std::vector<int> each_dims;
        for (int i=0; i<size-2; i++) {
            each_dims.push_back(args[i]);
        }
        Chl split_dim = (Chl)args[size-2];
        int head_size = (int)args[size-1];
        setup(*inputs[0], output, each_dims, split_dim, head_size);
    }
    void execute(vector<Tensor*> &output, vector<Tensor*> &inputs, vector<float> args) override{}
    void setup(Tensor &output, vector<Tensor *> &inputs, vector<float> args) override {}
    void execute(Tensor &output, vector<Tensor *> &inputs, vector<float> args) override {}  
};
} // namespace mllm
#endif // CPUTENSORFUNCTION_HPP
