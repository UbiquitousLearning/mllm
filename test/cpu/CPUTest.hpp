//
// Created by Xiang Li on 2023/10/14.
//

#ifndef MLLM_CPUTEST_HPP
#define MLLM_CPUTEST_HPP
#include "gtest/gtest.h"
#include "backends/cpu/CPUBackend.hpp"
#include "TestLoader.hpp"
#include "memory/SystemMemoryManager.hpp"
using namespace mllm;
#define COMPARE_TENSOR(...) ASSERT_TRUE(isSame(__VA_ARGS__))
#define TEST_LOAD(...) ASSERT_TRUE(loader.load(__VA_ARGS__)) << "TestLoader load failed"
#define TEST_WEIGHTS_LOAD(...) ASSERT_FALSE(op->load(__VA_ARGS__)) << "TestLoader load weights failed"
#define TEST_SETUP(...) ASSERT_FALSE(op->setUp(__VA_ARGS__))
#define TEST_RESHAPE(...) ASSERT_FALSE(op->reshape(__VA_ARGS__))
#define TEST_EXCUTE(...) ASSERT_FALSE(op->execute(__VA_ARGS__))
#define SETUP_OP(type_, ...)                       \
    auto op = new type_(bn_, #type_, __VA_ARGS__); \
    auto loader = TestLoader(::testing::UnitTest::GetInstance()->current_test_info()->name())
#define TENSOR(name_)                           \
    auto name_ = std::make_shared<Tensor>(bn_); \
    name_->setName(#name_);
#define SETUP_LOADER \
    auto loader = TestLoader(::testing::UnitTest::GetInstance()->current_test_info()->name())
#define PRINT_TENSOR_SHAPES(...)                                                                            \
    do {                                                                                                    \
        auto __tensors__ = {__VA_ARGS__};                                                                   \
        for (const auto &tensor : __tensors__) {                                                            \
            std::cout << "Tensor " << tensor->name() << ": [" << tensor->shapeString() << "]" << std::endl; \
        }                                                                                                   \
    } while (false)
class CPUTest : public ::testing::Test {
public:
    CPUTest() {
        mm_ = shared_ptr<MemoryManager>(new SystemMemoryManager());
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
    double eps = 0.000001;
    int flag = 0;
    if (b->ctype() == BCTHW) {
        a->setCtype(BCTHW);
        for (int i = 0; i < a->batch(); ++i) {
            for (int j = 0; j < a->channel(); ++j) {
                for (int k = 0; k < a->time(); ++k) {
                    for (int l = 0; l < a->height(); ++l) {
                        for (int m = 0; m < a->width(); ++m) {
                            double a_ = a->dataAt<float>(i, j, k, l, m);
                            double b_ = b->dataAt<float>(i, j, k, l, m);

                            //                     if ((a_ < b_) || (a_ > b_)) {
                            if ((abs(a_ - b_) / std::max(a_, b_)) > eps && !unstrict) {
                                std::cout << std::setprecision(8) << std::setiosflags(std::ios::fixed | std::ios::showpoint) << "a[" << i << "," << j << "," << k << "," << l << "," << m << "]: " << (double)a->dataAt<float>(i, j, k, l, m) << "!= b[" << i << "," << j << "," << k << "," << l << "," << m << "]: " << (double)b->dataAt<float>(i, j, k, l, m) << std::endl;
                                //                        return false;
                                std::cout << std::setprecision(8) << std::setiosflags(std::ios::fixed | std::ios::showpoint) << "Diff:" << abs(a_ - b_) / std::max(a_, b_) << std::endl;
                                flag += 1;
                                if (flag > 10) {
                                    return false;
                                }
                            }
                            if (unstrict) {
                                if (abs(a_ - b_) > 1e-4) {
                                    std::cout << std::setprecision(8) << std::setiosflags(std::ios::fixed | std::ios::showpoint) << "a[" << i << "," << j << "," << k << "," << l << "," << m << "]: " << (double)a->dataAt<float>(i, j, k, l, m) << "!= b[" << i << "," << j << "," << k << "," << l << "," << m << "]: " << (double)b->dataAt<float>(i, j, k, l, m) << std::endl;
                                    //                        return false;
                                    std::cout << std::setprecision(8) << std::setiosflags(std::ios::fixed | std::ios::showpoint) << "Diff:" << abs(a_ - b_) << std::endl;
                                    flag += 1;
                                    if (flag > 10) {
                                        return false;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < a->batch(); ++i) {
            for (int j = 0; j < a->head(); ++j) {
                for (int k = 0; k < a->sequence(); ++k) {
                    for (int l = 0; l < a->dimension(); ++l) {
                        double a_ = a->dataAt<float>({i, j, k, l});
                        double b_ = b->dataAt<float>({i, j, k, l});

                        //                     if ((a_ < b_) || (a_ > b_)) {
                        if ((abs(a_ - b_) / std::max(a_, b_)) > eps && !unstrict) {
                            std::cout << std::setprecision(8) << std::setiosflags(std::ios::fixed | std::ios::showpoint) << "a[" << i << "," << j << "," << k << "," << l << "]: " << (double)a->dataAt<float>(i, j, k, l) << "!= b[" << i << "," << j << "," << k << "," << l << "]: " << (double)b->dataAt<float>(i, j, k, l) << std::endl;
                            //                        return false;
                            std::cout << std::setprecision(8) << std::setiosflags(std::ios::fixed | std::ios::showpoint) << "Diff:" << abs(a_ - b_) / std::max(a_, b_) << std::endl;
                            flag += 1;
                            if (flag > 10) {
                                return false;
                            }
                        }
                        if (unstrict) {
                            if (abs(a_ - b_) > 1e-3) {
                                std::cout << std::setprecision(8) << std::setiosflags(std::ios::fixed | std::ios::showpoint) << "a[" << i << "," << j << "," << k << "," << l << "]: " << (double)a->dataAt<float>(i, j, k, l) << "!= b[" << i << "," << j << "," << k << "," << l << "]: " << (double)b->dataAt<float>(i, j, k, l) << std::endl;
                                //                        return false;
                                std::cout << std::setprecision(8) << std::setiosflags(std::ios::fixed | std::ios::showpoint) << "Diff:" << abs(a_ - b_) << std::endl;
                                flag += 1;
                                if (flag > 10) {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return flag == 0;
}
static bool isSame(shared_ptr<Tensor> a, shared_ptr<Tensor> b, bool unstrict = false) {
    return isSame(a.get(), b.get(), unstrict);
}
#endif // MLLM_CPUTEST_HPP
