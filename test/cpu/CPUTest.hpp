//
// Created by 咸的鱼 on 2023/10/14.
//

#ifndef MLLM_CPUTEST_HPP
#define MLLM_CPUTEST_HPP
#include "gtest/gtest.h"
#include "backends/cpu/CPUBackend.hpp"
using namespace mllm;
class CPUTest: public ::testing::Test {
public:
    CPUTest() {
        mm_ = shared_ptr<MemoryManager>(new MemoryManager());
    }
    ~CPUTest() {
        std::cout << "~CPUTest()" << std::endl;
    }
    virtual void SetUp() {
        bn_ = new CPUBackend(mm_);
        std::cout << "SetUp()" << std::endl;
    }
    virtual void TearDown() {
        delete bn_;
        std::cout << "TearDown()" << std::endl;
    }
protected:
    Backend* bn_;
    shared_ptr<MemoryManager> mm_;

};

#endif // MLLM_CPUTEST_HPP
