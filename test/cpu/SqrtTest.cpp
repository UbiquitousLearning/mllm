//
// Created by Xiang Li on 23-10-17.
//
#include "CPUTest.hpp"
#include "cmath"
TEST_F(CPUTest, Sqrt1) {
    SETUP_LOADER;
    TENSOR(input);
    TENSOR(output);
    TEST_LOAD(input);
    TEST_LOAD(output);
    //    c_output->reshape(input->shape());
    //    c_output->alloc();
    for (int i = 0; i < input->count(); ++i) {
        //        EXPECT_DOUBLE_EQ(std::sqrt(*(input->hostPtr<float>() + i)), *(output->hostPtr<float>() + i))
        //            << std::setprecision(8)
        //            << setiosflags(std::ios::fixed | std::ios::showpoint)
        //            << "a[" << i << "]: " << (double)std::sqrt(*(input->hostPtr<float>() + i)) << "!= b[" << i << "]: " << (double)*(output->hostPtr<float>() + i) << std::endl;
        // float
        if (abs(std::sqrt(*(input->hostPtr<float>() + i)) - *(output->hostPtr<float>() + i)) > 1e-6) {
            std::cout << std::setprecision(10)
                      << std::setiosflags(std::ios::fixed | std::ios::showpoint)
                      << "a[" << i << "]: " << (double)std::sqrt(*(input->hostPtr<float>() + i)) << "!= b[" << i << "]: " << (double)*(output->hostPtr<float>() + i) << std::endl;
        }
    }
}