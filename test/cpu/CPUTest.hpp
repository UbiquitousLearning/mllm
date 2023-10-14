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
static bool isSame(Tensor *a, Tensor *b) {
    if (a->legacyShape(0) != b->legacyShape(0) || a->legacyShape(1) != b->legacyShape(1) || a->legacyShape(2) != b->legacyShape(2) || a->legacyShape(3) != b->legacyShape(3)) {
        std::cout << "Shape a: " << a->shapeString() << " Shape b: " << b->shapeString() << std::endl;
        return false;
    }
    for (int i = 0; i < a->legacyShape(0); ++i) {
        for (int j = 0; j < a->legacyShape(1); ++j) {
            for (int k = 0; k < a->legacyShape(2); ++k) {
                for (int l = 0; l < a->legacyShape(3); ++l) {
                    if (a->dataAt<float>({i, j, k, l}) != b->dataAt<float>({i, j, k, l})) {
                        std::cout << "a[" << i << "," << j << "," << k << "," << l << "]" << a->dataAt<float>(i, j, k, l) << "!= b[" << i << "," << j << "," << k << "," << l << "]" << b->dataAt<float>(i, j, k, l) << std::endl;
                        return false;
                    }
                }
            }
        }
    }
}

#endif // MLLM_CPUTEST_HPP
