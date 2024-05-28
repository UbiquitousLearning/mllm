#include "StrassenMatmul.hpp"
#include "Types.hpp"
#include <cstddef>

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

    // TODO: save a,b,c in a stack?

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
    if (src1->dtype() == MLLM_TYPE_I8) { // int8 matmul for smoothquant
        switch (src1->dtype()) {
        case MLLM_TYPE_I8: { // q * k
            // NSHD
            float scale1_value = scale1 / 127.0;
            scale1_value = roundf(scale1_value * 10000) / 10000;

            float scale2_value = scale2 / 127.0;
            scale2_value = roundf(scale2_value * 10000) / 10000;

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
        return NOT_SUPPORT;
    }
    return MLLM_NO_ERROR;
}

ErrorCode StrassenMatmul::generateStrassenMatmul(int m, int k, int n, Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, bool transpose0, bool transpose1, float scale1, float scale2, int depth) {
    auto subM = m / 2;
    auto subK = k / 2;
    auto subN = n / 2;
    auto remainM = m - subM * 2;
    auto remainN = n - subN * 2;

    if (depth >= max_depth_ || k <= 64 || k % 2 != 0) {
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
    B11->reshape(src1->batch(), src1->head(), subK, subN);
    B11->deepCopyFrom(src1, false, {0, 0, 0, 0});
    auto *B12 = new Tensor();
    B12->setCtype(src1->ctype());
    B12->reshape(src1->batch(), src1->head(), subK, subN);
    B12->deepCopyFrom(src1, false, {0, 0, 0, subN});
    auto *B21 = new Tensor();
    B21->setCtype(src1->ctype());
    B21->reshape(src1->batch(), src1->head(), subK, subN);
    B21->deepCopyFrom(src1, false, {0, 0, subK, 0});
    auto *B22 = new Tensor();
    B22->setCtype(src1->ctype());
    B22->reshape(src1->batch(), src1->head(), subK, subN);

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
    X->reshape(src0->batch(), src0->head(), subM, subK);
    X->alloc();
    auto *Y = new Tensor();
    Y->setDtype(src1->dtype());
    Y->reshape(src1->batch(), src1->head(), subK, subN);
    Y->alloc();

    {
        // S3=A11-A21, T3=B22-B12, P7=S3*T3
        auto f = [A11, A21, B22, B12, X, Y, subM, subK, subN](int tId){
            // sub multi thread
        };
        functions_.push_back(std::make_pair(f, 0));
        auto code = generateStrassenMatmul(subM, subK, subN, A11, B11, X, false, nullptr, transpose0, transpose1, scale1, scale2, depth + 1);
        if(code != MLLM_NO_ERROR){
            return code;
        }
    }
    {
        // S1=A21+A22, T1=B12-B11, P5=S1T1
    }
    {
        // S2=S1-A11, T2=B22-T1, P6=S2T2
    }
    {
        // S4=A12-S2, P3=S4*B22, P1=A11*B11
    }
    {
        // U2=P1+P6, U3=U2+P7, U4=U2+P5, U7=U3+P5
        // U5=U4+P3, T4=T2-B21, P4=A22*T4
    }
    {
        // U6=U3-P4, P2=A12*B21, U1=P1+P2
    }
    if (remainM != 0) {
        generateTrivalMatmul(src0, src1, dst, false, NULL, transpose0, transpose1);
    }
    if (remainN != 0) {
        generateTrivalMatmul(src0, src1, dst, false, NULL, transpose0, transpose1);
    }
    return MLLM_NO_ERROR;
}

} // namespace mllm