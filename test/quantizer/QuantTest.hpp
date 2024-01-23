//
// Created by Xiang Li on 23-11-1.
//

#ifndef MLLM_QUANTTEST_HPP
#define MLLM_QUANTTEST_HPP
#include <gtest/gtest.h>
#include "ParamWriter.hpp"
#include "ParamLoader.hpp"
#include "Types.hpp"
static bool compare_eq(block_q4_0 *a, block_q4_0 *b) {
    if (a->d != b->d) {
        return false;
    }
    for (int i = 0; i < QK4_0 / 2; i++) {
        if (a->qs[i] != b->qs[i]) {
            return false;
        }
    }
    return true;
}

class QuantTest : public ::testing::Test {
public:
    QuantTest() {
     }
    ~QuantTest() {
     }
    virtual void SetUp() {

    }
    virtual void TearDown() {

    }

protected:
};

#endif // MLLM_QUANTTEST_HPP
