//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUTENSORFUNCTION_HPP
#define CPUTENSORFUNCTION_HPP
#include "Tensor.hpp"
#include "compute/Matmul.hpp"

namespace mllm {
class Tensor;

inline int thread_count = 4;
class CPUmmFunction {
    static void tranTensorChl(Tensor &input) {
        assert(input.ctype() == BSHD);
        auto b = input.batch();
        auto h = input.head();
        auto d = input.dimension();
        auto s = input.sequence();
        auto ori_seq_idx = input.chls_[SEQUENCE];
        auto ori_head_idx = input.chls_[HEAD];
        auto ori_dim_idx = input.chls_[DIMENSION];
        input.chls_[HEAD] = ori_seq_idx;
        input.chls_[DIMENSION] = ori_head_idx;
        input.chls_[SEQUENCE] = ori_dim_idx;
        input.changeCtype();
        input.reshape(b, h, s, d);
        input.transed() = true;
        input.undiffusion() = false;
    }
public:
    static void reshape(Tensor &input0, Tensor &input1, Tensor &output) {
        if(input1.chls_[SEQUENCE] != 3) {
            tranTensorChl(input1);
        }
        assert(input0.dimension() == input1.sequence());
        if (input0.dimension() == input1.sequence()) {
            output.reshape(input0.batch(), input0.head(), input0.sequence(), input1.dimension());
        }
    }
    static void setup(Tensor &input0, Tensor &input1, Tensor &output) {
        output.setDtype(input0.dtype());
        output.alloc();
    }
    static void execute(Tensor &input0, Tensor &input1, Tensor &output) {
        bool isSame = std::equal(input0.chls_.begin(), input0.chls_.end(), input1.chls_.begin());
        assert(input0.dtype() == MLLM_TYPE_F32);
        switch (input1.dtype()) {
        case MLLM_TYPE_F32: {
            mat_mul_fp32(&input0, &input1, &output, false, nullptr, false, isSame, thread_count);
            break;
        }
        case MLLM_TYPE_F16: {
            mat_mul_fp32_fp16(&input0, &input1, &output, false, nullptr, false, isSame, thread_count);
            break;
        }
        default:
            break;
        }
    }
};

class CPUnormFunction {
public:
    static void reshape(Tensor &input,  Tensor &output, int L_n) {
        output.reshape(input.batch(), input.head(), input.sequence(), input.dimension());
    }
    static void setup(Tensor &input,  Tensor &output, int L_n) {
        output.setDtype(input.dtype());
        output.alloc();
    }
    static void execute(Tensor &input,  Tensor &output, int L_n) {
        for (int h = 0; h < input.head(); h++) {
            for (int n = 0; n < input.batch(); n++) {
                for (int s = 0; s < input.sequence(); s++) {
                    if (L_n == 2) {
                        float sum_of_squares = 0.0f;
                        for (int d = 0; d < input.dimension(); ++d) {
                            sum_of_squares += input.dataAt<float>(n, h, s,d) * input.dataAt<float>(n, h, s,d);
                        }
                        float l2_norm = std::sqrt(sum_of_squares);
#pragma omp parallel for num_threads(thread_count)
                        for (int d = 0; d < input.dimension(); d++) {
                            output.setDataAt<float>(n, h, s,d, l2_norm);
                        }
                    } else {
                        float sum_of_abs_values = 0.0f;
                        for (int d = 0; d < input.dimension(); ++d) {
                            sum_of_abs_values += std::abs(input.dataAt<float>(n, h, s,d));
                        }
#pragma omp parallel for num_threads(thread_count)
                        for (int d = 0; d < input.dimension(); d++) {
                            output.setDataAt<float>(n, h, s,d, sum_of_abs_values);
                        }

                    }
                }
            }
        }
    }
    
};


class CPUbinaryFunction {
public:
    static void reshape(Tensor &input, Tensor &output) {
        output.reshape(input.batch(), input.head(), input.sequence(), input.dimension());
    }
    static void setup(Tensor &input, Tensor &output) {
        output.setDtype(input.dtype());
        output.alloc();
    }
    template <typename Func>
    static void execute(Tensor &input, Tensor &output, Func operation, float data) {
        if (input.masterTensor() == nullptr && output.masterTensor() == nullptr && input.ctype() == output.ctype()) {
#pragma omp parallel for num_threads(thread_count)
            for (int is = 0; is < input.batch() * input.head() * input.sequence() * input.dimension(); ++is) {
                output.hostPtr<float>()[is] = operation(input.hostPtr<float>()[is], data);
            }
        } else {
            for (int n = 0; n < input.batch(); ++n) {
                for (int c = 0; c < input.head(); ++c) {
                    for (int h = 0; h < input.sequence(); ++h) {
#pragma omp parallel for num_threads(thread_count)
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

class CPUbinaryTwoFunction {
public:
    static void reshape(Tensor &input0, Tensor &input1, Tensor &output) {
        output.reshape(std::max(input0.batch(), input1.batch()), input0.head(), input0.sequence(), input0.dimension());
    }
    static void setup(Tensor &input0, Tensor &input1, Tensor &output) {
        output.setDtype(input0.dtype());
        output.alloc();
    }
    template <typename Func>
    static void execute(Tensor &input0, Tensor &input1, Tensor &output, Func operation) {
        int batch_ = std::max(input0.batch(), input1.batch());
        if (input0.masterTensor() == nullptr && output.masterTensor() == nullptr && input0.ctype() == output.ctype()) {
            for (int n = 0; n < batch_; ++n) {
                auto n_0 = std::min(n, input0.batch() - 1);
                auto n_1 = std::min(n, input1.batch() - 1);
#pragma omp parallel for num_threads(thread_count)
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
#pragma omp parallel for num_threads(thread_count)
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
class CPUmeanFunction {
public:
    static void reshape(Tensor &input, Tensor &output, Chl axis) {
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
    }
    static void setup(Tensor &input, Tensor &output, Chl axis) {
        output.setDtype(input.dtype());
        output.alloc();
    }
    static void execute(Tensor &input, Tensor &output, Chl axis) {
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
};

class CPUviewFunction {
public:
    static void reshape(Tensor &input, Tensor &output, int b, int h, int s, int d) {
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
    }
    static void setup(Tensor &input, Tensor &output, int b, int h, int s, int d) {
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
    static void execute(Tensor &input, Tensor &output) {
    }
};

class CPUflattenFunction {
public:
    static void reshape(Tensor &input, Tensor &output, Chl axis_start, Chl axis_end) {
        int dim_b = input.batch();
        int dim_h = 0;
        int dim_s = 0;
        int dim_d = 0;
        /*
        if (input.ctype() == BSHD) {
            dim_h = input.head();
            dim_s = input.sequence();
            dim_d = input.dimension();
            if (axis_start == BATCH & axis_end == SEQUENCE) {
                // data_dims = {-1, HEAD, BATCH + SEQUENCE, DIMENSION};
                dim_b = 1;
                dim_s = input.sequence() * input.batch();
            } else if (axis_start == HEAD & axis_end == SEQUENCE) {
                // data_dims = {BATCH, -1, HEAD + SEQUENCE, DIMENSION};
                dim_h = 1;
                dim_s = input.sequence() * input.head();
            } else if (axis_start == HEAD & axis_end == DIMENSION) {
                // data_dims = {BATCH, HEAD, -1, SEQUENCE + DIMENSION};
                dim_h = 1;
                dim_d = input.dimension() * input.head();
            } else {
                std::cout << "ERROR:  flatten  " << axis_start << "&" << axis_end << std::endl;
            }
        } else if (input.ctype() == BHDS) {
            dim_h = input.head();
            dim_s = input.dimension();
            dim_d = input.sequence();
            if (axis_start == BATCH & axis_end == SEQUENCE) {
                // data_dims = {-1, HEAD, BATCH + SEQUENCE, DIMENSION};
                dim_b = 1;
                dim_s = dim_s * input.batch();
            } else if (axis_start == HEAD & axis_end == SEQUENCE) {
                // data_dims = {BATCH, -1, HEAD + SEQUENCE, DIMENSION};
                dim_h = 1;
                dim_s = dim_s * input.head();
            } else if (axis_start == HEAD & axis_end == DIMENSION) {
                // data_dims = {BATCH, HEAD, -1, SEQUENCE + DIMENSION};
                dim_h = 1;
                dim_d = dim_d * input.head();
            } else {
                std::cout << "ERROR:  flatten  " << axis_start << "&" << axis_end << std::endl;
            }
        } else if (input.ctype() == BDHS) {
            dim_h = input.head();
            dim_s = input.sequence();
            dim_d = input.dimension();
            if (axis_start == HEAD & axis_end == SEQUENCE) {
                dim_h = 1;
                dim_s = input.sequence() * input.head();
            }
        }else {
            if (axis_start == TIME & axis_end == CHANNLE) {
                // data_dims = {BATCH, -1, TIME + HEIGHT + WIDTH, CHANNLE};
                if (input.ctype() == BTHWC) {
                    dim_h = 1;
                    dim_s = input.time() * input.height() * input.width();
                    dim_d = input.channel();
                } else if (input.ctype() == BCTHW) {
                    dim_h = 1;
                    dim_s = input.time() * input.height() * input.channel();
                    dim_d = input.width();
                } else {
                    std::cout << "ERROR: flatten  " << axis_start << "&" << axis_end << std::endl;
                }
            }
        }*/
        if(input.shape().size() == 4) {
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
            }else {
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
        assert(dim_d+dim_s+dim_h > 0);
        output.reshape(dim_b, dim_h, dim_s, dim_d);
    }
    static void setup(Tensor &input, Tensor &output, Chl axis_start, Chl axis_end) {
        if (   (axis_start == TIME & axis_end == WIDTH && input.ctype()==BCTHW)
            || (axis_start == CHANNLE & axis_end == HEIGHT && input.ctype()==BWCTH)
            || (axis_start == HEIGHT & axis_end == CHANNLE && input.ctype()==BTHWC)
            || (axis_start == BATCH & axis_end == SEQUENCE && input.ctype()!=BCTHW)
            || (axis_start == HEAD & axis_end == SEQUENCE && input.ctype()==BSHD)
            || (axis_start == HEAD & axis_end == SEQUENCE && input.ctype()==BHDS)
            || (axis_start == HEAD & axis_end == SEQUENCE && input.ctype()==BDHS)
            || (axis_start == HEAD & axis_end == DIMENSION && input.ctype()==BSHD)
            || (axis_start == HEAD & axis_end == DIMENSION && input.ctype()==BHDS)
            || (axis_start == HEAD & axis_end == SEQUENCE && input.ctype()==BDSH)
        ){
            if(input.masterTensor() == nullptr) {
                input.free();
            }
            output.setDtype(input.dtype());
            output.alloc();
            input.deepCopyFrom(output, false);
        }else {
            std::cout<<"[TODO]Tensor.Flatten not support!!!!"<<std::endl;
        }
    }
    static void execute(Tensor &input, Tensor &output) {
    }
};

class CPUclipFunction {
public:
    static void reshape(Tensor &input, Tensor &output, vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
        // reshape
        int dim_b = input.batch();
        int dim_h = input.head();
        int dim_s = input.sequence();
        int dim_d = input.dimension();
        std::vector<std::pair<std::vector<int>, int*>> data = {{b, &dim_b}, {h, &dim_h}, {s, &dim_s}, {d, &dim_d}};
        for (auto& pair : data) {
            if (pair.first.size() == 2) {
                *pair.second = pair.first[1] - pair.first[0];
            } else if (pair.first.size() == 1) {
                *pair.second = 1;
            }
        }
        output.reshape(dim_b, dim_h, dim_s, dim_d);
    }
    static void setup(Tensor &input, Tensor &output, vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
        output.setDtype(input.dtype());
        output.alloc();
    }
    static void execute(Tensor &input, Tensor &output, vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
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
        }else {
            std::cout<<"[TODO]Tensor.CLip not support!!!!"<<std::endl;
        }
    }
};

class CPUclipaxisFunction {
public:
    static void reshape(Tensor &input, Tensor &output,Chl axis, vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
        // reshape
        int dim_b = input.batch();
        int dim_h = input.head();
        int dim_s = input.sequence();
        int dim_d = input.dimension();

        /*
        std::vector<std::pair<std::vector<int>, int*>> data = {{b, &dim_b}, {h, &dim_h}, {s, &dim_s}, {d, &dim_d}};
        for (auto& pair : data) {
            if (pair.first.size() > 0) {
                *pair.second = 1;
            }
        }
        */
        switch (axis) {
        case BATCH: {
            std::vector<std::pair<std::vector<int>, int*>> data = {{h, &dim_h}, {s, &dim_s}, {d, &dim_d}};
            for (auto& pair : data) {
                if (pair.first.size() > 0) {
                    *pair.second = 1;
                }
            }
            break;
        }
        case HEAD: {
            std::vector<std::pair<std::vector<int>, int*>> data = {{b, &dim_b}, {s, &dim_s}, {d, &dim_d}};
            for (auto& pair : data) {
                if (pair.first.size() > 0) {
                    *pair.second = 1;
                }
            }
            break;
        }
        case SEQUENCE: {
            std::vector<std::pair<std::vector<int>, int*>> data = {{b, &dim_b}, {h, &dim_h}, {d, &dim_d}};
            for (auto& pair : data) {
                if (pair.first.size() > 0) {
                    *pair.second = 1;
                }
            }
            break;
        }
        case DIMENSION: {
            std::vector<std::pair<std::vector<int>, int*>> data = {{b, &dim_b}, {h, &dim_h}, {s, &dim_s}};
            for (auto& pair : data) {
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
    }
    static void setup(Tensor &input, Tensor &output, Chl axis, vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
        output.setDtype(input.dtype());
        output.alloc();
    }
    static void execute(Tensor &input, Tensor &output, Chl axis, vector<int> b, vector<int> h, vector<int> s, vector<int> d) {
        if (axis == BATCH) {
            if(s.size()>0) {
                for (int i = 0; i < s.size(); ++i) {
                    auto seq_idx = s[i];
                    memcpy(output.hostPtr<float>() + output.offset(i, 0, 0, 0),
                           input.hostPtr<float>() + input.offset(i, 0, seq_idx, 0),
                           input.head() * 1 * input.dimension() * sizeof(float));
                }
            }
        } else {
            std::cout<<"[TODO]Tensor.CLip not support!!!!"<<std::endl;
        }
    }
};

class CPUcatFunction {
public:
    static void reshape(vector<Tensor *>inputs, Tensor &output, Chl axis, int expd_batch_, int expd_batch_input_idx) {
        int dim_b = expd_batch_;
        int dim_h = inputs[0]->head();
        int dim_s = inputs[0]->sequence();
        int dim_d = inputs[0]->dimension();
        int sizes[] = {0, 0, 0, 0};
        Chl axes[] = {BATCH, HEAD, SEQUENCE, DIMENSION};
        int* dims[] = {&dim_b, &dim_h, &dim_s, &dim_d};
        for (int i = 0; i < 4; i++) {
            if (axis == axes[i]) {
                for (auto input : inputs) {
                    sizes[i] += (i == 0) ? input->batch() : (i == 1) ? input->head() : (i == 2) ? input->sequence() : input->dimension();
                }
                *dims[i] = sizes[i];
                break;
            }
        }
        output.reshape(dim_b, dim_h, dim_s, dim_d);
    }
    static void setup(vector<Tensor *>inputs, Tensor &output, Chl axis, int expd_batch_, int expd_batch_input_idx) {
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
                    cseq += inputs[idx-1]->sequence();
                }
                inputs[idx]->deepCopyFrom(output, false, {cbatch, chead, cseq, cdim}); // b,h,s,d
            }
        }
    }
    static void execute(vector<Tensor *>inputs, Tensor &output, Chl axis, int expd_batch_, int expd_batch_input_idx) {
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
};

} // namespace mllm
#endif // CPUTENSORFUNCTION_HPP
