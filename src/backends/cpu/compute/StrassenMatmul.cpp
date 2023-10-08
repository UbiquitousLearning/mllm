//
// Created by ey on 23-9-28.
//

#include "StrassenMatmul.hpp"
#include <iostream>
#include <vector>

using namespace std;

namespace mllm {

// 矩阵相加
void matrixAdd(shared_ptr<Tensor> &A, shared_ptr<Tensor> &B, shared_ptr<Tensor> &C, vector<int> A_offsets, vector<int> B_offsets, vector<int> C_offsets, int batch = 0) {
    CHECK_EQ(A->batch(), B->batch());
    CHECK_EQ(A->hidden(), B->hidden());
    CHECK_EQ(A->seqLen(), B->seqLen());
    CHECK_EQ(A->width(), B->width());
    int out_batch = batch;
    if (A_offsets.empty()) {
        A_offsets = {0, A->hidden(), 0, A->seqLen()};
    }
    if (B_offsets.empty()) {
        B_offsets = {0, B->hidden(), 0, B->seqLen()};
    }
    if (!A_offsets.empty() && !B_offsets.empty()) {
        CHECK_EQ(A_offsets[1] - A_offsets[0], B_offsets[1] - B_offsets[0]);
        CHECK_EQ(A_offsets[3] - A_offsets[2], B_offsets[3] - B_offsets[2]);
    }
    int hidden = A_offsets[1] - A_offsets[0];
    int seqLen = A_offsets[3] - A_offsets[2];
    if (C_offsets.empty()) {
        C_offsets = {0, hidden, 0, seqLen};
        C->reshape(1, hidden, seqLen, 1);
        out_batch = 0;
    } else {
        CHECK_EQ(hidden, C->hidden());
        CHECK_EQ(seqLen, C->seqLen());
    }
    if (!C->allocted()) {
        C->alloc();
    }
    for (int i = 0; i < hidden; i++) {
        for (int j = 0; j < seqLen; j++) {
            C->setDataAt<float>(out_batch, i + C_offsets[0], j + C_offsets[2], 0,
                                A->dataAt<float>(batch, i + A_offsets[0], j + A_offsets[2], 0)
                                    + B->dataAt<float>(batch, i + B_offsets[0], j + B_offsets[2], 0));
        }
    }
}
// 矩阵相减
void matrixSub(shared_ptr<Tensor> &A, shared_ptr<Tensor> &B, shared_ptr<Tensor> &C, vector<int> A_offsets, vector<int> B_offsets, vector<int> C_offsets, int batch = 0) {
    CHECK_EQ(A->batch(), B->batch());
    CHECK_EQ(A->hidden(), B->hidden());
    CHECK_EQ(A->seqLen(), B->seqLen());
    CHECK_EQ(A->width(), B->width());
    int out_batch = batch;
    if (A_offsets.empty()) {
        A_offsets = {0, A->hidden(), 0, A->seqLen()};
    }
    if (B_offsets.empty()) {
        B_offsets = {0, B->hidden(), 0, B->seqLen()};
    }
    if (!A_offsets.empty() && !B_offsets.empty()) {
        CHECK_EQ(A_offsets[1] - A_offsets[0], B_offsets[1] - B_offsets[0]);
        CHECK_EQ(A_offsets[3] - A_offsets[2], B_offsets[3] - B_offsets[2]);
    }
    int hidden = A_offsets[1] - A_offsets[0];
    int seqLen = A_offsets[3] - A_offsets[2];
    if (C_offsets.empty()) {
        C_offsets = {0, hidden, 0, seqLen};
        C->reshape(1, hidden, seqLen, 1);
        out_batch = 0;
    } else {
        CHECK_EQ(hidden, C->hidden());
        CHECK_EQ(seqLen, C->seqLen());
    }
    if (!C->allocted()) {
        C->alloc();
    }
    for (int i = 0; i < hidden; i++) {
        for (int j = 0; j < seqLen; j++) {
            C->setDataAt<float>(out_batch, i + C_offsets[0], j + C_offsets[2], 0,
                                A->dataAt<float>(batch, i + A_offsets[0], j + A_offsets[2], 0)
                                    - B->dataAt<float>(batch, i + B_offsets[0], j + B_offsets[2], 0));
        }
    }
}
// 将矩阵划分成四个子矩阵
void splitMatrix(shared_ptr<Tensor> &A, vector<int> &A11_offsets, vector<int> &A12_offsets, vector<int> &A21_offsets, vector<int> &A22_offsets) {
    int A_hidden = A->hidden();
    int A_seqLen = A->seqLen();
    A11_offsets = {0, A_hidden / 2, 0, A_seqLen / 2};
    A12_offsets = {0, A_hidden / 2, A_seqLen / 2, A_seqLen};
    A21_offsets = {A_hidden / 2, A_hidden, 0, A_seqLen / 2};
    A22_offsets = {A_hidden / 2, A_hidden, A_seqLen / 2, A_seqLen};
}

void strassenMatMul(shared_ptr<Tensor> &A, shared_ptr<Tensor> &B, shared_ptr<Tensor> &C, vector<int> A_offsets, vector<int> B_offsets, vector<int> C_offsets, int batch) {
    if (A_offsets.empty()) {
        A_offsets = {0, A->hidden(), 0, A->seqLen()};
    }
    if (B_offsets.empty()) {
        B_offsets = {0, B->hidden(), 0, B->seqLen()};
    }
    if (C_offsets.empty()) {
        C_offsets = {0, C->hidden(), 0, C->seqLen()};
    }
    if (A_offsets[1] - A_offsets[0] <= 1 || A_offsets[3] - A_offsets[2] <= 1) {
        int M = A_offsets[1] - A_offsets[0];
        int K = A_offsets[3] - A_offsets[2];
        int N = B_offsets[3] - B_offsets[2];
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float value = 0;
                for (int k = 0; k < K; k++) {
                    value += A->dataAt<float>(batch, m + A_offsets[0], k + A_offsets[3], 0)
                             * B->dataAt<float>(batch, k + B_offsets[0], n + B_offsets[3], 0);
                }
                C->setDataAt<float>(batch, m + C_offsets[0], n + C_offsets[3], 0, value);
            }
        }
        return;
    }
    vector<int> A11_offsets;
    vector<int> A12_offsets;
    vector<int> A21_offsets;
    vector<int> A22_offsets;
    vector<int> B11_offsets;
    vector<int> B12_offsets;
    vector<int> B21_offsets;
    vector<int> B22_offsets;
    splitMatrix(A, A11_offsets, A12_offsets, A21_offsets, A22_offsets);
    splitMatrix(B, B11_offsets, B12_offsets, B21_offsets, B22_offsets);

    // S1=A21+A22
    shared_ptr<Tensor> S1(new Tensor());
    matrixAdd(A, A, S1, A21_offsets, A22_offsets, {}, batch);
    // S2=S1-A11
    shared_ptr<Tensor> S2(new Tensor());
    matrixSub(S1, A, S2, {}, A11_offsets, {}, batch);
    // S3=A11-A21
    shared_ptr<Tensor> S3(new Tensor());
    matrixSub(A, A, S3, A11_offsets, A21_offsets, {}, batch);
    // S4=A12-S2
    shared_ptr<Tensor> S4(new Tensor());
    matrixSub(A, S2, S4, A12_offsets, {}, {}, batch);
    // T1=B12-B11
    shared_ptr<Tensor> T1(new Tensor());
    matrixSub(B, B, T1, B12_offsets, B11_offsets, {}, batch);
    // T2=B22-T1
    shared_ptr<Tensor> T2(new Tensor());
    matrixSub(B, T1, T2, B22_offsets, {}, {}, batch);
    // T3=B22-B12
    shared_ptr<Tensor> T3(new Tensor());
    matrixSub(B, B, T3, B22_offsets, B12_offsets, {}, batch);
    // T4=T2-B21
    shared_ptr<Tensor> T4(new Tensor());
    matrixSub(T2, B, T4, {}, B21_offsets, {}, batch);

    // P1=A11*B11
    shared_ptr<Tensor> P1(new Tensor());
    P1->reshape(1, A11_offsets[1] - A11_offsets[0], B11_offsets[3] - B11_offsets[2], 1);
    P1->alloc();
    strassenMatMul(A, B, P1, A11_offsets, B11_offsets, {}, batch);
    // P2=A12*B21
    shared_ptr<Tensor> P2(new Tensor());
    P2->reshape(1, A12_offsets[1] - A12_offsets[0], B21_offsets[3] - B21_offsets[2], 1);
    P2->alloc();
    strassenMatMul(A, B, P2, A12_offsets, B21_offsets, {}, batch);
    // P3=S4*B22
    shared_ptr<Tensor> P3(new Tensor());
    P3->reshape(1, S4->hidden(), B22_offsets[3] - B22_offsets[2], 1);
    P3->alloc();
    strassenMatMul(S4, B, P3, {}, B22_offsets, {}, batch);
    // P4=A22*T4
    shared_ptr<Tensor> P4(new Tensor());
    P4->reshape(1, A22_offsets[1] - A22_offsets[0], T4->seqLen(), 1);
    P4->alloc();
    strassenMatMul(A, T4, P4, A22_offsets, {}, {}, batch);
    // P5=S1*T1
    shared_ptr<Tensor> P5(new Tensor());
    P5->reshape(1, S1->hidden(), T1->seqLen(), 1);
    P5->alloc();
    strassenMatMul(S1, T1, P5, {}, {}, {}, 0);
    // P6=S2*T2
    shared_ptr<Tensor> P6(new Tensor());
    P6->reshape(1, S2->hidden(), T2->seqLen(), 1);
    P6->alloc();
    strassenMatMul(S2, T2, P6, {}, {}, {}, 0);
    // P7=S3*T3
    shared_ptr<Tensor> P7(new Tensor());
    P7->reshape(1, S3->hidden(), T3->seqLen(), 1);
    P7->alloc();
    strassenMatMul(S3, T3, P7, {}, {}, {}, 0);

    // U1=P1+P2
    //    shared_ptr<Tensor> U1(new Tensor());
    //    matrixAdd(P1, P2, U1, {}, {}, {}, 0);
    vector<int> U1_offsets = {0, P1->hidden(), 0, P1->seqLen()};
    matrixAdd(P1, P2, C, {}, {}, U1_offsets, 0);
    // U2=U1+P6
    shared_ptr<Tensor> U2(new Tensor());
    //    matrixAdd(U1, P6, U2, {}, {}, {}, 0);
    matrixAdd(C, P6, U2, U1_offsets, {}, {}, 0);
    // U3=U2+P7
    shared_ptr<Tensor> U3(new Tensor());
    matrixAdd(U2, P7, U3, {}, {}, {}, 0);
    // U4=U2+P5
    shared_ptr<Tensor> U4(new Tensor());
    matrixAdd(U2, P5, U4, {}, {}, {}, 0);
    // U5=U4+P3
    //    shared_ptr<Tensor> U5(new Tensor());
    //    matrixAdd(U4, P3, U5, {}, {}, {}, 0);
    vector<int> U5_offsets = {0, U4->hidden(),  U1_offsets[3],  U1_offsets[3] + U4->seqLen()};
    matrixAdd(U4, P3, C, {}, {}, U5_offsets, 0);
    // U6=U3-P4
    //    shared_ptr<Tensor> U6(new Tensor());
    //    matrixSub(U3, P4, U6, {}, {}, {}, 0);
    vector<int> U6_offsets = {U1_offsets[1], U1_offsets[1] + U3->hidden(), 0, U3->seqLen()};
    matrixSub(U3, P4, C, {}, {}, U6_offsets, 0);
    // U7=U3+P5
    //    shared_ptr<Tensor> U7(new Tensor());
    //    matrixAdd(U3, P5, U7, {}, {}, {}, 0);
    vector<int> U7_offsets = {U1_offsets[1], U1_offsets[1] + U3->hidden(), U1_offsets[3], U1_offsets[3]+ U3->seqLen()};
    matrixAdd(U3, P5, C, {}, {}, U7_offsets, 0);
}
} // namespace mllm
