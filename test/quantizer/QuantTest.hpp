//
// Created by lx on 23-11-1.
//

#ifndef MLLM_QUANTTEST_HPP
#define MLLM_QUANTTEST_HPP
#include <gtest/gtest.h>
#include "ParamWriter.hpp"
#include "ParamLoader.hpp"
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
