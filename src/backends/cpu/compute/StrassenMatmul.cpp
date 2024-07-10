#include "StrassenMatmul.hpp"
#include "Types.hpp"
#include <cassert>
#include <cstddef>
#include <utility>

namespace mllm {

StrassenMatmul::StrassenMatmul(Backend *bn, int maxDepth, int threadCount) :
    backend_(bn), max_depth_(maxDepth), thread_count(threadCount) {
}

void StrassenMatmul::onReset() {
    functions_.clear();
}

ErrorCode StrassenMatmul::onReshape(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1, int scale1, int scale2) {
    const int M = transpose0 ? src0->dimension() : src0->sequence();
    const int K = transpose0 ? src0->sequence() : src0->dimension();
    const int N = transpose1 ? src1->sequence() : src1->dimension();

    return generateStrassenMatmul(M, K, N, src0, src1, dst, support_bias, bias, transpose0, transpose1, scale1, scale2);
}

ErrorCode StrassenMatmul::onExecute() {
    ThreadPool threadPool(thread_count);
    for (auto &f : functions_) {
        threadPool.enqueue(std::bind(f.first, f.second));
    }
    return MLLM_NO_ERROR;
}

ErrorCode StrassenMatmul::generateTrivalMatmul(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1, float scale1, float scale2) {
    auto f = [src0, src1, dst, support_bias, bias, transpose0, transpose1, scale1, scale2, this](int tId) {
        if (src1->dtype() == MLLM_TYPE_I8) { // int8 matmul for smoothquant
            switch (src1->dtype()) {
            case MLLM_TYPE_I8: { // q * k
                // NSHD
                float scale1_value = scale1 / 127.0;
                scale1_value = roundf(scale1_value * 10000) / 10000;

                float scale2_value = scale2 / 127.0;
                scale2_value = roundf(scale2_value * 10000) / 10000;

// std::cout << "------------- in i8 matmul" << std::endl;
// src0->printShape();
// src1->printShape();
// dst->printShape();
// std::cout << transpose0 << std::endl;
// std::cout << transpose1 << std::endl;
                mat_mul_i8(src0, src1, dst, false, nullptr, transpose0, transpose1, thread_count, scale1_value, scale2_value);
                break;
            }
            case MLLM_TYPE_F32: { // qk * v
                float scale_value = scale2 / 127.0;
                scale_value = roundf(scale_value * 10000) / 10000;

                mat_mul_fp32_i8(src0, src1, dst, false, nullptr, transpose0, transpose1, thread_count, scale_value);
                break;
            }
            default:
                break;
            }
        } else if (src0->dtype() == MLLM_TYPE_F32) { // cpu common matmul
            switch (src1->dtype()) {
            case MLLM_TYPE_F32: {
                mat_mul_fp32(src0, src1, dst, false, nullptr, transpose0, transpose1, thread_count);
                break;
            }
            case MLLM_TYPE_F16: {
                mat_mul_fp32_fp16(src0, src1, dst, false, nullptr, transpose0, transpose1, thread_count);
                break;
            }
            default:
                break;
            }
        } else {
            std::cerr << "Unsupported data type" << std::endl;
            exit(1);
        }
    };
    functions_.push_back(std::make_pair(f, 0));
    return MLLM_NO_ERROR;
}

void TensorSub(Tensor *src0, Tensor *src1, Tensor *dst, int thread_count) {
    assert(src0->dtype() == src1->dtype());
    int N = std::max(src0->batch(), src1->batch());
    int C = src0->head();
    int H = src0->sequence();
    int W = src0->dimension();

    if (src1->dtype() == MLLM_TYPE_I8) {
        for (int n = 0; n < N; ++n) {
            auto n_0 = std::min(n, src0->batch() - 1);
            auto n_1 = std::min(n, src1->batch() - 1);
            if (src0->masterTensor() == nullptr && src1->masterTensor() == nullptr && src0->ctype() == src1->ctype()) {
                auto copy_size = C * H * W;
                auto in0_ptr = src0->ptrAt<int8_t>(n_0, 0, 0, 0);
                auto in1_ptr = src1->ptrAt<int8_t>(n_1, 0, 0, 0);
                auto out_ptr = dst->ptrAt<int8_t>(n, 0, 0, 0);
#pragma omp parallel for num_threads(thread_count)
                for (int is = 0; is < copy_size; ++is) {
                    out_ptr[is] = in0_ptr[is] - in1_ptr[is];
                }
            } else {
                for (int c = 0; c < C; ++c) {
                    for (int h = 0; h < H; ++h) {
#pragma omp parallel for num_threads(thread_count)
                        for (int w = 0; w < W; ++w) {
                            dst->setDataAt<int8_t>(n, c, h, w, src0->dataAt<int8_t>(n_0, c, h, w) - src1->dataAt<int8_t>(n_1, c, h, w));
                        }
                    }
                }
            }
        }
    } else if (src1->dtype() == MLLM_TYPE_F32) {
        for (int n = 0; n < N; ++n) {
            auto n_0 = std::min(n, src0->batch() - 1);
            auto n_1 = std::min(n, src1->batch() - 1);
            if (src0->masterTensor() == nullptr && src1->masterTensor() == nullptr && src0->ctype() == src1->ctype()) {
                auto copy_size = C * H * W;
                auto in0_ptr = src0->ptrAt<float>(n_0, 0, 0, 0);
                auto in1_ptr = src1->ptrAt<float>(n_1, 0, 0, 0);
                auto out_ptr = dst->ptrAt<float>(n, 0, 0, 0);
#pragma omp parallel for num_threads(thread_count)
                for (int is = 0; is < copy_size; ++is) {
                    out_ptr[is] = in0_ptr[is] - in1_ptr[is];
                }
            } else {
                for (int c = 0; c < C; ++c) {
                    for (int h = 0; h < H; ++h) {
#pragma omp parallel for num_threads(thread_count)
                        for (int w = 0; w < W; ++w) {
                            dst->setDataAt<float>(n, c, h, w, src0->dataAt<float>(n_0, c, h, w) - src1->dataAt<float>(n_1, c, h, w));
                        }
                    }
                }
            }
        }
    } else {
        std::cerr << "Unsupported data type" << std::endl;
        exit(1);
    }
}

void TensorAdd(Tensor *src0, Tensor *src1, Tensor *dst, int thread_count) {
    assert(src0->dtype() == src1->dtype());
    int N = std::max(src0->batch(), src1->batch());
    int C = src0->head();
    int H = src0->sequence();
    int W = src0->dimension();

    if (src1->dtype() == MLLM_TYPE_I8) {
        for (int n = 0; n < N; ++n) {
            auto n_0 = std::min(n, src0->batch() - 1);
            auto n_1 = std::min(n, src1->batch() - 1);
            if (src0->masterTensor() == nullptr && src1->masterTensor() == nullptr && src0->ctype() == src1->ctype()) {
                auto copy_size = C * H * W;
                auto in0_ptr = src0->ptrAt<int8_t>(n_0, 0, 0, 0);
                auto in1_ptr = src1->ptrAt<int8_t>(n_1, 0, 0, 0);
                auto out_ptr = dst->ptrAt<int8_t>(n, 0, 0, 0);
#pragma omp parallel for num_threads(thread_count)
                for (int is = 0; is < copy_size; ++is) {
                    out_ptr[is] = in0_ptr[is] + in1_ptr[is];
                }
            } else {
                for (int c = 0; c < C; ++c) {
                    for (int h = 0; h < H; ++h) {
#pragma omp parallel for num_threads(thread_count)
                        for (int w = 0; w < W; ++w) {
                            dst->setDataAt<int8_t>(n, c, h, w, src0->dataAt<int8_t>(n_0, c, h, w) + src1->dataAt<int8_t>(n_1, c, h, w));
                        }
                    }
                }
            }
        }
    } else if (src1->dtype() == MLLM_TYPE_F32) {
        for (int n = 0; n < N; ++n) {
            auto n_0 = std::min(n, src0->batch() - 1);
            auto n_1 = std::min(n, src1->batch() - 1);
            if (src0->masterTensor() == nullptr && src1->masterTensor() == nullptr && src0->ctype() == src1->ctype()) {
                auto copy_size = C * H * W;
                auto in0_ptr = src0->ptrAt<float>(n_0, 0, 0, 0);
                auto in1_ptr = src1->ptrAt<float>(n_1, 0, 0, 0);
                auto out_ptr = dst->ptrAt<float>(n, 0, 0, 0);
#pragma omp parallel for num_threads(thread_count)
                for (int is = 0; is < copy_size; ++is) {
                    out_ptr[is] = in0_ptr[is] + in1_ptr[is];
                }
            } else {
                for (int c = 0; c < C; ++c) {
                    for (int h = 0; h < H; ++h) {
#pragma omp parallel for num_threads(thread_count)
                        for (int w = 0; w < W; ++w) {
                            dst->setDataAt<float>(n, c, h, w, src0->dataAt<float>(n_0, c, h, w) + src1->dataAt<float>(n_1, c, h, w));
                        }
                    }
                }
            }
        }
    } else {
        std::cerr << "Unsupported data type" << std::endl;
        exit(1);
    }
}

ErrorCode StrassenMatmul::generateStrassenMatmul(int m, int k, int n, Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1, float scale1, float scale2, int depth) {
    auto subM = m / 2;
    auto subK = k / 2;
    auto subN = n / 2;
    auto remainM = m - subM * 2;
    auto remainN = n - subN * 2;

    if (depth >= max_depth_ || k <= 16 || k % 2 != 0) {
        return generateTrivalMatmul(src0, src1, dst, support_bias, bias, transpose0, transpose1, scale1, scale2);
    }
    auto *A11 = new Tensor();
    A11->setCtype(src0->ctype());
    A11->reshape(src0->batch(), src0->head(), subM, subK);
    A11->deepCopyFrom(src0, false, {0, 0, 0, 0});
    auto *A12 = new Tensor();
    A12->setCtype(src0->ctype());
    A12->reshape(src0->batch(), src0->head(), subM, subK);
    A12->deepCopyFrom(src0, false, {0, 0, 0, subK});
    auto *A21 = new Tensor();
    A21->setCtype(src0->ctype());
    A21->reshape(src0->batch(), src0->head(), subM, subK);
    A21->deepCopyFrom(src0, false, {0, 0, subM, 0});
    auto *A22 = new Tensor();
    A22->setCtype(src0->ctype());
    A22->reshape(src0->batch(), src0->head(), subM, subK);
    A22->deepCopyFrom(src0, false, {0, 0, subM, subK});

    auto *B11 = new Tensor();
    B11->setCtype(src1->ctype());
    B11->reshape(src1->batch(), src1->head(), subN, subK);
    B11->deepCopyFrom(src1, false, {0, 0, 0, 0});
    auto *B12 = new Tensor();
    B12->setCtype(src1->ctype());
    B12->reshape(src1->batch(), src1->head(), subN, subK);
    B12->deepCopyFrom(src1, false, {0, 0, 0, subK});
    auto *B21 = new Tensor();
    B21->setCtype(src1->ctype());
    B21->reshape(src1->batch(), src1->head(), subN, subK);
    B21->deepCopyFrom(src1, false, {0, 0, subN, 0});
    auto *B22 = new Tensor();
    B22->setCtype(src1->ctype());
    B22->reshape(src1->batch(), src1->head(), subN, subK);
    B22->deepCopyFrom(src1, false, {0, 0, subN, subK});

    auto *C11 = new Tensor();
    C11->setCtype(dst->ctype());
    C11->reshape(dst->batch(), dst->head(), subM, subN);
    C11->deepCopyFrom(dst, false, {0, 0, 0, 0});
    auto *C12 = new Tensor();
    C12->setCtype(dst->ctype());
    C12->reshape(dst->batch(), dst->head(), subM, subN);
    C12->deepCopyFrom(dst, false, {0, 0, 0, subN});
    auto *C21 = new Tensor();
    C21->setCtype(dst->ctype());
    C21->reshape(dst->batch(), dst->head(), subM, subN);
    C21->deepCopyFrom(dst, false, {0, 0, subM, 0});
    auto *C22 = new Tensor();
    C22->setCtype(dst->ctype());
    C22->reshape(dst->batch(), dst->head(), subM, subN);
    C22->deepCopyFrom(dst, false, {0, 0, subM, subN});

    auto *X = new Tensor();
    X->setDtype(src0->dtype());
    X->setBackend(this->backend_);
    X->reshape(src0->batch(), src0->head(), subM, subK);
    X->alloc();
    auto *Y = new Tensor();
    Y->setBackend(this->backend_);
    Y->setDtype(src1->dtype());
    Y->reshape(src1->batch(), src1->head(), subN, subK);
    Y->alloc();
    auto *CX = new Tensor();
    CX->setBackend(this->backend_);
    CX->setDtype(dst->dtype());
    CX->reshape(dst->batch(), dst->head(), subM, subN);
    CX->alloc();

    {
        std::cout << "S3=A11-A21, T3=B22-B12, P7=S3*T3" << depth << std::endl;
        // S3=A11-A21, T3=B22-B12, P7=S3*T3
        auto f = [A11, A21, B22, B12, X, Y, subM, subK, subN, this](int tId) {
            // std::cout << "S3=A11-A21, T3=B22-B12, P7=S3*T3" << std::endl;
            TensorSub(A11, A21, X, this->thread_count);
            TensorSub(B22, B12, Y, this->thread_count);
        };
        functions_.push_back(std::make_pair(f, 0));
        auto code = generateStrassenMatmul(subM, subK, subN, X, Y, C21, false, nullptr, transpose0, transpose1, scale1, scale2, depth + 1);
        if (code != MLLM_NO_ERROR) {
            return code;
        }
    }
    {
        std::cout << "S1=A21+A22, T1=B12-B11, P5=S1T1" << depth << std::endl;
        // S1=A21+A22, T1=B12-B11, P5=S1T1
        auto f = [A21, A22, B12, B11, X, Y, subM, subK, subN, this](int tId) {
            // std::cout << "S1=A21+A22, T1=B12-B11, P5=S1T1" << std::endl;
            TensorAdd(A21, A22, X, this->thread_count);
            TensorSub(B12, B11, Y, this->thread_count);
        };
        functions_.push_back(std::make_pair(f, 0));
        auto code = generateStrassenMatmul(subM, subK, subN, X, Y, C22, false, nullptr, transpose0, transpose1, scale1, scale2, depth + 1);
        if (code != MLLM_NO_ERROR) {
            return code;
        }
    }
    {
        std::cout << "S2=S1-A11, T2=B22-T1, P6=S2T2" << depth << std::endl;
        // S2=S1-A11, T2=B22-T1, P6=S2T2
        auto f = [A11, X, B22, Y, subM, subK, subN, this](int tId) {
            // std::cout << "S2=S1-A11, T2=B22-T1, P6=S2T2" << std::endl;
            TensorSub(X, A11, X, this->thread_count);
            TensorSub(B22, Y, Y, this->thread_count);
        };
        functions_.push_back(std::make_pair(f, 0));
        auto code = generateStrassenMatmul(subM, subK, subN, X, Y, C12, false, nullptr, transpose0, transpose1, scale1, scale2, depth + 1);
        if (code != MLLM_NO_ERROR) {
            return code;
        }
    }
    {
        std::cout << "S4=A12-S2, P3=S4*B22, P1=A11*B11" << depth << std::endl;
        // S4=A12-S2, P3=S4*B22, P1=A11*B11
        auto f = [A12, X, B22, Y, subM, subK, subN, this](int tId) {
            // std::cout << "S4=A12-S2, P3=S4*B22, P1=A11*B11" << std::endl;
            TensorSub(A12, X, X, this->thread_count);
        };
        functions_.push_back(std::make_pair(f, 0));
        auto code = generateStrassenMatmul(subM, subK, subN, X, B22, C11, false, nullptr, transpose0, transpose1, scale1, scale2, depth + 1);
        if (code != MLLM_NO_ERROR) {
            return code;
        }
        code = generateStrassenMatmul(subM, subK, subN, A11, B11, CX, false, nullptr, transpose0, transpose1, scale1, scale2, depth + 1);
        if (code != MLLM_NO_ERROR) {
            return code;
        }
    }
    {
        std::cout << "U2=P1+P6, U3=U2+P7, U4=U2+P5, U7=U3+P5" << depth << std::endl;
        // U2=P1+P6, U3=U2+P7, U4=U2+P5, U7=U3+P5
        // U5=U4+P3, T4=T2-B21, P4=A22*T4
        auto f = [B21, C11, C12, C21, C22, CX, X, Y, subM, subK, subN, this](int tId) {
            // std::cout << "U2=P1+P6, U3=U2+P7, U4=U2+P5, U7=U3+P5" << std::endl;
            TensorAdd(CX, C12, C12, this->thread_count);
            TensorAdd(C12, C21, C21, this->thread_count);
            TensorAdd(C12, C22, C12, this->thread_count);
            TensorAdd(C21, C22, C22, this->thread_count);
            TensorAdd(C12, C11, C11, this->thread_count);
            TensorSub(Y, B21, Y, this->thread_count);
        };
        functions_.push_back(std::make_pair(f, 0));
        auto code = generateStrassenMatmul(subM, subK, subN, A22, Y, C11, false, nullptr, transpose0, transpose1, scale1, scale2, depth + 1);
    }
    {
        std::cout << "U6=U3-P4, P2=A12*B21, U1=P1+P2" << depth << std::endl;
        // U6=U3-P4, P2=A12*B21, U1=P1+P2
        auto f0 = [C11, C12, C21, C22, X, Y, subM, subK, subN, this](int tId) {
            // std::cout << "U6=U3-P4, P2=A12*B21, U1=P1+P2" << std::endl;
            TensorSub(C21, C11, C11, this->thread_count);
        };
        functions_.push_back(std::make_pair(f0, 0));
        auto code = generateStrassenMatmul(subM, subK, subN, A12, B21, C11, false, nullptr, transpose0, transpose1, scale1, scale2, depth + 1);
        if (code != MLLM_NO_ERROR) {
            return code;
        }
        auto f1 = [C11, CX, subM, subK, subN, this](int tId) {
            // std::cout << "U6=U3-P4, P2=A12*B21, U1=P1+P2 ADD" << std::endl;
            TensorAdd(C11, CX, C11, this->thread_count);
        };
        functions_.push_back(std::make_pair(f1, 0));
    }
    // if (remainM != 0) {
    //     generateTrivalMatmul(src0, src1, dst, false, NULL, transpose0, transpose1);
    // }
    // if (remainN != 0) {
    //     generateTrivalMatmul(src0, src1, dst, false, NULL, transpose0, transpose1);
    // }
    return MLLM_NO_ERROR;
}

ThreadPool::ThreadPool(size_t num_threads) :
    stop_(false) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(
            [this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->condition_.wait(lock,
                                              [this] { return this->stop_ || !this->tasks_.empty(); });
                        if (this->stop_ && this->tasks_.empty())
                            return;
                        task = std::move(this->tasks_.front());
                        this->tasks_.pop();
                    }
                    task();
                }
            });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (std::thread &worker : workers_)
        worker.join();
}

void ThreadPool::enqueue(std::function<void()> f) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_)
            throw std::runtime_error("enqueue on stopped ThreadPool");
        tasks_.emplace(std::move(f));
    }
    condition_.notify_one();
}

} // namespace mllm