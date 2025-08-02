// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"

#include "KernelTestHelper.hpp"

#define MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(OpName, EnumDType, CDType, __Op)               \
  bool OpName(const std::vector<mllm::Tensor::shape_t>& shapes) {                        \
    using mllm::kCPU;                                                                    \
    using mllm::EnumDType;                                                               \
    using mllm::Tensor;                                                                  \
    for (auto& s : shapes) {                                                             \
      Tensor a = Tensor::random(s, -16, 16, EnumDType, kCPU);                            \
      Tensor b = Tensor::random(s, -16, 16, EnumDType, kCPU);                            \
      Tensor ref_c = Tensor::zeros(s, EnumDType, kCPU);                                  \
      {                                                                                  \
        auto a_ptr = a.ptr<mllm::CDType>();                                              \
        auto b_ptr = b.ptr<mllm::CDType>();                                              \
        auto c_ptr = ref_c.ptr<mllm::CDType>();                                          \
        auto num_elements = a.numel();                                                   \
        for (size_t i = 0; i < num_elements; i++) { c_ptr[i] = a_ptr[i] __Op b_ptr[i]; } \
      }                                                                                  \
      auto c = a __Op b;                                                                 \
      auto result = mllm::test::allClose(c, ref_c);                                      \
      if (!result) {                                                                     \
        mllm::print(c);                                                                  \
        mllm::print(ref_c);                                                              \
        mllm::print(result);                                                             \
        return false;                                                                    \
      }                                                                                  \
    }                                                                                    \
    return true;                                                                         \
  }

class ElementwiseKernelTest : public KernelTest {
 public:
  ElementwiseKernelTest() = default;
  ~ElementwiseKernelTest() override = default;

  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(AddFloat32Test, kFloat32, mllm_fp32_t, +)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(AddFloat16Test, kFloat16, mllm_fp16_t, +)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(AddInt8Test, kInt8, mllm_int8_t, +)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(AddInt16Test, kInt16, mllm_int16_t, +)
  MLLM_CPU_KERNEL_TEST_GEN_EW_TESTS(AddInt32Test, kInt32, mllm_int32_t, +)
};
