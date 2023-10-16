//
// Created by ey on 23-9-28.
//

#ifndef MLLM_STRASSENMATMUL_H
#define MLLM_STRASSENMATMUL_H
#include "Backend.hpp"
#include "Tensor.hpp"
namespace mllm {
class StrassenMatmul {
public:
    StrassenMatmul(Backend *bn, string opName, bool multiThread, int maxDepth);
    //    StrassenMatmul();
    virtual ~StrassenMatmul();

    /*
     It's assume that:
     P = core->pack
     A is a matrix where each element is a (P,1) vector : [l/P], e, P
     B is a matrix where each element is a (hP,1) vector : h, l, hP
     inputs[0] is the transpose of A: AT, inputs[1] is the transpose of B: BT
     outputs[0] is the transpose of C: CT
     C is a matrix where each element is a (P,1) vector, the same as A : [h/P], e, P

    if (inputs.size() > 2) {
        inputs[2] is origin CO: CT
        CO can be the same same as C or broadcast in lenght(1): hC4, e, P or hC4, 1, P
    }
    Compute: C = alpha * AB + beta * CO , alpha must be 1.0f

        postParameters:
        0: alpha
        1: beta
        2: min
        3: max

        if (postParameters.empty()) {
            alpha = 1.0f
            beta = 0.0f;
            min = -FLT_MAX
            max = FLT_MAX
        }
    */
    ErrorCode encode(const std::vector<shared_ptr<Tensor>> &inputs, const std::vector<shared_ptr<Tensor>> &outputs, const std::vector<float> &postParameters = {}, int l = 0, int h = 0);

    ErrorCode encode(int e, int l, int h, int as, int bs, int cs, const uint8_t *AT, const uint8_t *BT, uint8_t *CT, bool useBias, const uint8_t *Bias = nullptr, const std::vector<float> &postParameters = {});

    void execute(const uint8_t *AT = nullptr, const uint8_t *BT = nullptr, const uint8_t *COT = nullptr, uint8_t *CT = nullptr);

    Backend *backend() const {
        return backend_;
    }

private:
    Backend *backend_;
};

void strassenMatMul(shared_ptr<Tensor> &A, shared_ptr<Tensor> &B, shared_ptr<Tensor> &C, vector<int> A_offsets, vector<int> B_offsets, vector<int> C_offsets);
} // namespace mllm
#endif // MLLM_STRASSENMATMUL_H
