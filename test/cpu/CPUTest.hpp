//
// Created by 咸的鱼 on 2023/10/14.
//

#ifndef MLLM_CPUTEST_HPP
#define MLLM_CPUTEST_HPP
#include "gtest/gtest.h"
#include "backends/cpu/CPUBackend.hpp"
#include "TestLoader.hpp"
using namespace mllm;
#define COMPARE_TENSOR(...) EXPECT_TRUE(isSame(__VA_ARGS__))
#define TEST_LOAD(...) EXPECT_TRUE(loader.load(__VA_ARGS__))
#define TEST_SETUP(...) EXPECT_FALSE(op->setUp(__VA_ARGS__))
#define TEST_RESHAPE(...) EXPECT_FALSE(op->reshape(__VA_ARGS__))
#define TEST_EXCUTE(...) EXPECT_FALSE(op->execute(__VA_ARGS__))
#define SETUP_OP(type_, ...)                       \
    auto op = new type_(bn_, #type_, __VA_ARGS__); \
    auto loader = TestLoader(::testing::UnitTest::GetInstance()->current_test_info()->name())
#define TENSOR(name_)                           \
    auto name_ = std::make_shared<Tensor>(bn_); \
    name_->setName(#name_);
#define SETUP_LOADER \
    auto loader = TestLoader(::testing::UnitTest::GetInstance()->current_test_info()->name())
class CPUTest : public ::testing::Test {
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
    Backend *bn_;
    shared_ptr<MemoryManager> mm_;
};
static bool isSame(Tensor *a, Tensor *b, bool unstrict = false) {
    if (a->legacyShape(0) != b->legacyShape(0) || a->legacyShape(1) != b->legacyShape(1) || a->legacyShape(2) != b->legacyShape(2) || a->legacyShape(3) != b->legacyShape(3)) {
        std::cout << "Shape a: " << a->shapeString() << " Shape b: " << b->shapeString() << std::endl;
        return false;
    }
    double eps = 0.0000001;
    if (unstrict) {
        eps = 1e-5;
    }
    for (int i = 0; i < a->legacyShape(0); ++i) {
        for (int j = 0; j < a->legacyShape(1); ++j) {
            for (int k = 0; k < a->legacyShape(2); ++k) {
                for (int l = 0; l < a->legacyShape(3); ++l) {
                    double a_ = a->dataAt<float>({i, j, k, l});
                    double b_ = b->dataAt<float>({i, j, k, l});
                    // if ((a_ < b_) || (a_ > b_)) {
                    if (abs(a_ - b_) / std::max(a_, b_) > eps) {
                        std::cout << std::setprecision(8) << setiosflags(std::ios::fixed | std::ios::showpoint) << "a[" << i << "," << j << "," << k << "," << l << "]: " << (double)a->dataAt<float>(i, j, k, l) << "!= b[" << i << "," << j << "," << k << "," << l << "]: " << (double)b->dataAt<float>(i, j, k, l) << std::endl;
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

#endif // MLLM_CPUTEST_HPP
